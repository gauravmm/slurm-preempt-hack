import argparse
import logging
import re  # Now you have two problems.
import os
from io import open_code
from os import kill
from pathlib import Path
from pprint import pformat, pprint
from subprocess import PIPE, Popen, TimeoutExpired

from flask import Flask
from flask_restful import Api, Resource, abort, reqparse

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__file__)

COMMENT_FLAG = "PREEMPTIBLE" # What to look for in the comment 
USERNAME = os.environ["USER"]

class JobSelectError(RuntimeError):
    pass

def priority_earliest_first(j):
    parts = [int(p) for p in re.split('[-:]', j["TIME"])] + [int(j["JOBID"])] # JobID is the tie-breaker
    parts = tuple(([0]*4 + parts)[-5:]) # Zero-pad to four places
    return parts # The tuple is the priority

delim = "|"
def parse_job(open_str, priority_func=priority_earliest_first):
    # Split this into lines
    lines = open_str.split("\n")

    # There must be a header
    assert len(lines) >= 1
    header = lines.pop(0).split(delim)

    assert all(j in header for j in ["JOBID", "TIME", "COMMENT", "NODELIST", "STATE"])

    if len(lines) == 0:
        raise JobSelectError("No jobs currently running.")

    # Transform all jobs to a dictionary:
    jobs = [{k: v.strip() for k, v in zip(header, l.split(delim, maxsplit=len(header)))} for l in lines if l.strip()]

    # Select only jobs that are preemptible:
    jobs = [j for j in jobs if COMMENT_FLAG in j["COMMENT"]]

    if len(jobs) == 0:
        raise JobSelectError("No preemptible jobs currently running; there is nothing to do here.")

    # Identify the jobs still pending:
    jobs_pending = [j for j in jobs if j["STATE"]=="PENDING"]
    jobs_running = [j for j in jobs if j["STATE"]=="RUNNING"]

    jobs_other = [j for j in jobs if j["STATE"] not in ["RUNNING", "PENDING", "SPECIAL_EXIT"]]
    if len(jobs_other):
        logger.warn(f"Jobs in unknown state: {pformat(jobs_other)}")

    # Sort the jobs by cancellation priority, from first to last:
    jobs_running = sorted(jobs_running, key=priority_func)

    return jobs_running, jobs_pending

def run_command(cmd, dry_run=False):
    logger.debug(f"run_command({cmd})")
    if dry_run:
        return {"note": "DRY_RUN"}

    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True)
    returncode = None
    try:
        stdout, stderr = p.communicate(timeout=5)
        returncode = p.returncode
    except TimeoutExpired:
        p.kill()
        stdout, stderr = p.communicate()
        returncode = "TIMEOUT"

    rv = {"stdout": stdout, "stderr": stderr, "returncode": returncode}
    logger.debug(f"returned with: {pformat(rv)}")
    return rv

def pause_jobs(js, **kwargs):
    return run_command(["scontrol", "hold"] + [j["JOBID"] for j in js if j["JOBID"]], **kwargs)

def kill_jobs(js, **kwargs):
    return run_command(["scontrol", "requeuehold", "Incomplete"] + [j["JOBID"] for j in js if j["JOBID"]], **kwargs)

def read_jobs():
    rv = run_command(['squeue', '-u', USERNAME, '--format', r'%A|%j|%M|%N|%T|%k'])
    if rv["returncode"] != 0:
        raise JobSelectError(f"Command to list jobs terminated with error code {rv['returncode']}")
    return parse_job(rv["stdout"])

def main(args):
    assert USERNAME.strip()
    logger.info(f"Looking for jobs from {USERNAME}")
    if args.test:
        logger.info(f"Testing on {args.test}")
        jobs_running, jobs_pending = parse_job(args.test.read_text())
        if len(jobs_pending):
            pause_jobs(jobs_pending, dry_run=True)
        if len(jobs_running):
            kill_jobs(jobs_running[0:1], dry_run=True)
    else:
        listener(args)

def delete_job_request(job_ids=None, **kwargs):
    jobs_running, jobs_pending = read_jobs()
    output = [f"Attempting to preempt {'specified jobs' if job_ids else 'some job'} from <{USERNAME}>:"]

    jobs_to_kill = []
    if job_ids is not None:
        jobs_to_kill = [j for j in jobs_running if j["JOBID"] in job_ids]
    else:
        if len(jobs_running):
            jobs_to_kill = jobs_running[0:1]

    if len(jobs_pending):
        output += [f"Paused {len(jobs_pending)} jobs."]
        pause_jobs(jobs_pending, **kwargs)

    if len(jobs_to_kill):
        kill_jobs(jobs_to_kill, **kwargs)
        output += ["Killed jobs: " + ", ".join(f"{j['JOBID']} on {j['NODELIST']}" for j in jobs_to_kill) + ""]
    else:
        output += [f"(No matching jobs to kill. {len(jobs_running)} preemptible jobs still running.)"]

    return output

def listener(args):
    app = Flask(__name__)
    api = Api(app, default_mediatype="text/plain")

    @api.representation('text/plain')
    def text(data, code, headers):
        if isinstance(data, list):
            data = "\n".join(data)
        elif isinstance(data, str):
            pass
        else:
            data = pformat(data)

        return app.make_response(data + "\n")

    class PreemptibleJobs(Resource):
        representations = {'text/plain': text}
        def get(self):
            """Get the state of all jobs."""
            jobs_running, jobs_pending = read_jobs()

            if len(jobs_running):
                rv = ["These jobs will be killed, in this order:",
                    "\t" + ", ".join(f"{j['JOBID']} on {j['NODELIST']}" for j in jobs_running)]
            else:
                rv = ["No preemptible jobs running."]

            rv+= ["",
                "These jobs will be paused:",
                "\t" + ", ".join(f"{j['JOBID']}" for j in jobs_pending),
                "",
                "Send this address a DELETE request to kill the first job on this list",
                "You can kill a specific job by sending a DELETE request to `/jobs/<job_id>`",
                "(Any DELETE request also pauses all pending jobs automatically.)"]

            return rv

        def delete(self):
            """Stop the top job automatically"""
            return delete_job_request(dry_run=args.dry_run)

    class SpecificPreemptibleJob(Resource):
        representations = {'text/plain': text}
        def get(self, job_id):
            """Get the state of a particular job."""
            jobs_running, jobs_pending = read_jobs()
            matching_jobs = [j for j in jobs_running if int(j['JOBID']) == int(job_id)]

            if len(matching_jobs):
                j = matching_jobs.pop()
                rv = [f"Job {j['JOBID']} can be killed:",
                    pformat(j),
                    "",
                    f"Send this address a DELETE request to kill job {j['JOBID']}",
                    f"(All {len(jobs_pending)} pending jobs will be paused automatically.)"]

                return rv

            else:
                rv = [f"Job {job_id} cannot be found.",
                f"Send /jobs a DELETE request to automatically select a job to kill."]

        def delete(self, job_id):
            """Stop the specified job"""
            return delete_job_request([job_id], dry_run=args.dry_run)

    api.add_resource(PreemptibleJobs, '/jobs')
    api.add_resource(SpecificPreemptibleJob, '/jobs/<job_id>')

    logger.info("STARTED")
    app.run(port="2222", host="localhost")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Requeue preemptible jobs for SLURM")
    parser.add_argument("--test", type=Path, default=None, help="test the job parser and halt")
    parser.add_argument("--dry-run", action="store_true", help="run the server without interacting with SLURM")
    main(parser.parse_args())
