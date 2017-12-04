import logging
import os
import sys
import subprocess

from github import Github

logger = logging.getLogger(__file__)


class GitCommandError(Exception):
    """A git command has failed"""


class GitClient:
    def __init__(self, config: dict):
        self.root = config['ROOT']
        self.new_version_name = config['NEW_VERSION_NAME']
        self.github_org = config['GITHUB_ORG']
        self.project_name = config['PROJECT_NAME']
        self.origin = Github(config['GITHUB_TOKEN']).get_organization(
            self.github_org,
        ).get_repo(
            self.project_name,
        )

    def execute_command(self, git_command: list):
        """Execute a git command"""
        from utils import run_subprocess
        try:
            return run_subprocess(
                ['git'] + git_command,
                cwd=self.root,
            )
        except subprocess.CalledProcessError as exc:
            logger.exception(
                'Failed git command=%s, output=%s',
                git_command,
                sys.exc_info()[1].stdout,
            )
            raise GitCommandError()

    def create_tag(self):
        """Create a git tag"""
        self.execute_command(
            ['tag', '-s', 'v' + self.new_version_name, '-m', 'v' + self.new_version_name,]
        )

    def create_ref(self):
        self.origin.create_git_ref(
            'refs/tags/{}'.format('v' + self.new_version_name),
            self.get_rev_hash('master'),  # TODO: this will have to be something else for hotfixes
        )

    def create_release(self, release_notes: str):
        self.origin.create_git_release(
            tag='v' + self.new_version_name,
            name='{} - Version {}'.format(self.project_name, self.new_version_name),
            message=release_notes,
            draft=False,
            prerelease=False,
        )

    def get_rev_hash(self, ref: str) -> str:
        """Get the SHA1 hash of a particular git ref"""
        return self.execute_command(
            ['rev-parse', ref]
        ).stdout.strip()  # Get rid of the newline character at the end

    def merge_and_create_release_commit(self, release_notes: str, changelog_path: str):
        """Create a release commit based on origin/develop and merge it to master"""
        from utils import update_changelog_file
        for git_command in [
                # TODO: deal with merge conflicts in an interactive way
                ['fetch', '-p'],
                ['checkout', '-B', 'release-{}'.format(self.new_version_name), 'origin/develop'],
        ]:
            self.execute_command(git_command)

        update_changelog_file(
            changelog_path,
            release_notes,
            logger,
        )

        for git_command in [
                ['add', changelog_path],
                ['commit', '-m', 'Release {}'.format(self.new_version_name)],
                ['checkout', '-B', 'master', 'origin/master'],
                ['merge', '--no-ff', '--no-edit', 'release-{}'.format(self.new_version_name)],
                ['push', 'origin', 'master'],
        ]:
            self.execute_command(git_command)

    def update_develop(self):
        """Merge master branch to develop"""
        for git_command in [
                ['fetch', '-p'],
                ['checkout', '-B', 'develop', 'origin/develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/master'],
                ['push', 'origin', 'develop'],
        ]:
            self.execute_command(git_command)

    @property
    def release_url(self) -> str:
        return 'https://github.com/{github_org}/{project_name}/releases/tag/{new_version_name}'.format(
            github_org=self.github_org,
            project_name=self.project_name,
            new_version_name='v' + self.new_version_name,
        )
