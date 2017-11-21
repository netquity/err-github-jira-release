import os
import sys
import subprocess

import utils


class GitCommandError(Exception):
    """A git command has failed"""


class GitClient:
    def __init__(self, project_root: str, log: 'logging.Logger'):
        self.root = project_root
        self.log = log

    def execute_command(self, git_command: list):
        """Execute a git command"""
        try:
            return utils.run_subprocess(
                ['git'] + git_command,
                cwd=self.root,
            )
        except subprocess.CalledProcessError as exc:
            self.log.exception(
                'Failed git command=%s, output=%s',
                git_command,
                sys.exc_info()[1].stdout,
            )
            raise GitCommandError()

    def create_tag(self, version_number: str):
        """Create a git tag"""
        self.execute_command(
            ['tag', '-s', 'v' + version_number, '-m', 'v' + version_number,]
        )

    def get_rev_hash(self, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        return self.execute_command(
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end

    def merge_and_create_release_commit(self, version_number: str, release_notes: str, changelog_path: str):
        """Create a release commit based on origin/develop and merge it to master"""
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', 'release-{}'.format(version_number), 'origin/develop'],
        ]:
            self.execute_command(git_command)

        utils.update_changelog_file(
            changelog_path,
            release_notes,
            self.log,
        )

        for git_command in [
                ['add', changelog_path],
                ['commit', '-m', 'Release {}'.format(version_number)],
                ['checkout', '-B', 'master', 'origin/master'],
                ['merge', '--no-ff', '--no-edit', 'release-{}'.format(version_number)],
                ['push', 'origin', 'master'],
        ]:
            self.execute_command(git_command)

    def merge_master_to_develop(self):
        """Merge master branch to develop"""
        for git_command in [
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/master'],
                ['push', 'origin', 'develop'],
        ]:
            self.execute_command(git_command)
