import json
import os
import shutil
from tqdm import tqdm

from settings.settings import *


def tracker_file(name):
    return os.path.join(TRACKER_FOLDER, name + '.json')


def job_folder(name, job):
    return os.path.join(JOB_RUNS_FOLDER, name, job)


class JobTracker:
    NOT_STARTED = 'not_started'
    STARTED = 'started'
    COMPLETED = 'completed'

    def __init__(self, name):
        self.name = name
        self.location = tracker_file(name)
        self.jobs = self._create_or_read()

    def _create_or_read(self):
        if os.path.exists(self.location):
            print("Reading from existing location")
            with open(self.location) as f:
                return json.load(f)
        else:
            print("Creating tracker anew")
            return {}

    def _update_file(self):
        """ Saves current tracker to disk """
        with open(self.location, 'w') as f:
            json.dump(self.jobs, f)

    def create_jobs(self, job_list: list):
        """Adds an immutable list of jobs to this tracker"""
        if bool(self.jobs):
            print("Jobs have already been assigned")
        else:
            print("Creating new jobs")
            self.jobs = {job: self.NOT_STARTED for job in job_list}
            self._update_file()

            # also create one folder for each job in the JOB_RUNS_FOLDER
            # will throw an error if they already exist
            for job in job_list:
                os.makedirs(job_folder(self.name, job), exist_ok=False)
        return True

    def mark_as_started(self, job):
        """Marks that a specific job has started; saves to disk"""
        self.jobs[job] = self.STARTED
        self._update_file()
        return True

    def mark_as_completed(self, job):
        """Marks that a specific job has ended; saves to disk"""
        self.jobs[job] = self.COMPLETED
        self._update_file()
        return True

    def read_to_do_jobs(self) -> list:
        return [job for job, status in self.jobs.items() if status == self.NOT_STARTED]

    def read_unfinished_jobs(self) -> list:
        return [job for job, status in self.jobs.items() if status == self.STARTED]

    def reset_unfinished_jobs(self):
        import glob
        """
        Does two things:
        1. For all job folders which are as STARTED, keep them but delete their contents
        2. Mark these jobs as NOT_STARTED
        """
        for job, status in self.jobs.items():
            if status == self.STARTED:
                # 1. delete contents of subfolder
                files = glob.glob(os.path.join(JOB_RUNS_FOLDER, job, "*"))
                for f in files:
                    os.remove(f)
                assert not glob.glob(os.path.join(JOB_RUNS_FOLDER, job, "*")), 'Files were not deleted upon reset'

                # 2. make job as not_started
                self.jobs[job] = self.NOT_STARTED

        self._update_file()

    def reset(self):

        cont = input("This will delete all folders and models. Are you sure? [y/n]")
        if cont not in ('n', 'no', 'N', 'NO'):

            # Delete all folders
            for job in tqdm(self.jobs.keys()):
                path = job_folder(self.name, job)
                if os.path.exists(path):
                    shutil.rmtree(path)

            # Delete current tracker
            os.remove(tracker_file(self.name))

    def report(self):
        total_jobs = len(self.jobs)
        unfinished = len(self.read_unfinished_jobs())
        not_started = len(self.read_to_do_jobs())
        finished = total_jobs - unfinished - not_started
        print(f"Job batch: {self.name}  -- Total jobs: {total_jobs}")
        print(f"  > Finished:    {finished} ({round(100 * finished / total_jobs)} %)")
        print(f"  > Not started: {not_started} ({round(100 * not_started / total_jobs)} %)")
        print(f"  > Unfinished:  {unfinished} ({round(100 * unfinished / total_jobs)} %)")
