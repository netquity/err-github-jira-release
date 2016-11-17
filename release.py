# coding: utf-8
import datetime
import logging
import subprocess

from errbot import BotPlugin, arg_botcmd
from errbot.utils import ValidationException, recurse_check_structure
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__file__)

try:
    from jira import JIRA, JIRAError
    from github import Github
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")


class JIRAVersionError(Exception):
    """Could not find the given JIRA version resource."""


class NoJIRAIssuesFoundError(Exception):
    """No JIRA issues match the given search parameters."""


class Release(BotPlugin):  # pylint:disable=too-many-ancestors
    """Perform version releases between JIRA and GitHub.

    For a given JIRA project key, produce a tagged release on GitHub. Produce the version by taking all of JIRA tickets
    under the given project key where status=closed and there is no `Fix Version`. Inspect all of the tickets in that
    set to determine the highest `Release Type` value, where choices are [Major,Minor,Patch], in accordance with
    semver.org.  Use the highest found `Release Type` value to decide the version bump from the last released version
    for that project.

    Example:
        Project Key: FOO
        Last Release: 1.2.3
        Tickets closed since last release:
            FOO-100: Minor
            FOO-101: Patch
            FOO-102: Major  <-- the highest `Release Type` since the last release

        FOO-102 determines that the release is a major version bump; the result is FOO v2.0.0. FOO-{100,101,102} all
        get updated with `Fix Version = 2.0.0`. The release notes are added to the CHANGELOG.md file and to the tag
        created at GitHub.

    Note: Currently does not support hotfix releases; only standard releases from origin/develop to origin/master are
    supported.
    """

    def activate(self):
        if not self.config:
            # Don't allow activation until we are configured
            message = 'Release is not configured, please do so.'
            self.log.info(message)
            self.warn_admins(message)
            return

        self.jira_client = self.get_jira_client()  # pylint:disable=attribute-defined-outside-init
        self.github_client = self.get_github_client()  # pylint:disable=attribute-defined-outside-init
        super().activate()

    def get_configuration_template(self) -> str:
        return {
            'REPOS_ROOT': '/home/web/repos/',
            'JIRA_URL': None,
            'JIRA_USER': None,
            'JIRA_PASS': None,
            'GITHUB_TOKEN': None,
            'projects': {
                'some-project': {  # Name of the project in GitHub
                    'jira_key': 'PRJ',
                    'repo_url': 'git@github.com:netquity/some-project.git',
                    'github_org': 'netquity',
                },
            },
            'TEMPLATE_DIR': '/home/web/templates/',
            'changelog_path': '{}/CHANGELOG.md',
        }

    def check_configuration(self, configuration: 'typing.Mapping') -> None:
        """Allow for the `projects` key to have a variable number of definitions."""
        # Remove the `projects` key from both the template and the configuration and then test them separately
        try:
            config_template = self.get_configuration_template().copy()
            projects_template = config_template.pop('projects')
            projects_config = configuration.pop('projects')  # Might fail
        except KeyError:
            raise ValidationException(
                'Your configuration must include a projects key with at least one project configured.'
            )

        recurse_check_structure(config_template, configuration)

        # Check that each project configuration matches the template
        for k, v in projects_config.items():
            recurse_check_structure(projects_template['some-project'], v)

    @arg_botcmd('--project-key', dest='project_key', type=str.upper, required=True)
    def version(
            self,
            msg: 'errbot.backends.base.Message',
            project_key: str,
    ) -> str:
        """Perform a version release to GitHub using issues from JIRA."""
        # Check out latest
        # TODO: check validity of given version number
        try:
            project_name = self.jira_client.project(project_key).name
            project_root = self.get_project_root(project_name)

            jira_previous_version = self.get_jira_latest_version(project_key)
            jira_new_version = self.jira_client.create_version(
                Release.bump_version(
                    jira_previous_version.name,
                    self.get_jira_release_type(project_key)
                ),
                project=project_key,
                released=True,
                releaseDate=datetime.datetime.now().date(),
            )
            self.set_jira_fix_version(
                project_key,
                jira_new_version.name,
            )
            release_notes = self.get_jira_release_notes(jira_new_version)
        except JIRAError:
            exc_message = 'Unable to complete JIRA request for project_key=%s. Versioning aborted.' % project_key

            try:
                jira_new_version.delete()  # Remove version from issues it's attached to
            except JIRAError:
                exc_message = (
                    'Unable to complete JIRA request for project_key={} and unable to clean up new version={}'.format(
                        project_key,
                        jira_new_version.name,
                    )
                )

            self.log.exception(
                exc_message,
            )
            return exc_message

        self.update_changelog_file(project_root, release_notes)
        commit_hash = self.git_merge_and_create_release_commit(project_root, jira_new_version.name)

        self.github_client.get_organization(
            self.config['projects'][project_name]['github_org'],
        ).get_repo(
            project_name,
        ).create_git_tag_and_release(
            tag=jira_new_version.name,
            tag_message='v' + jira_new_version.name,
            release_name='{} - Version {}'.format(project_name, jira_new_version.name),
            release_message=release_notes,
            object=commit_hash,
            type='commit',
            # tagger=github.GithubObject.NotSet,  # TODO: add some author information
            draft=False,
            prerelease=False,
        )

        Release.git_merge_master_to_develop(project_root)

    def update_changelog_file(self, project_root: str, release_notes: str):
        """Prepend the given release notes to CHANGELOG.md."""
        # TODO: exceptions
        changelog_filename = self.config['changelog_path'].format(project_root)
        try:
            with open(changelog_filename, 'r') as changelog:
                original_contents = changelog.read()
            with open(changelog_filename, 'w') as changelog:
                changelog.write(release_notes + "\n" + original_contents)
        except OSError as exc:
            self.log.exception('An unknown error occurred while updating the changelog file.')
            raise exc

    def git_merge_and_create_release_commit(self, project_root: str, version_number: str) -> str:
        """Wrap subprocess calls with some project-specific defaults.

        :param project_root:
        :param version_number:
        :return: Release commit hash.
        """
        try:
            for argv in [
                    # TODO: deal with merge conflicts in an interactive way
                    ['fetch', '-p'],
                    ['checkout', 'master'],
                    ['checkout', '--merge', '-b', 'release-{}'.format(version_number), 'origin/develop'],
                    ['add', self.config['changelog_path'].format(project_root)],
                    ['commit', '-m', 'Release {}'.format(version_number)],
                    ['checkout', 'master'],
                    ['reset', '--hard', 'origin/master'],
                    # TODO: make sure we're not losing hotfix stuff or other
                    ['merge', '-X', 'theirs', '--no-ff', '--no-edit', 'release-{}'.format(version_number)],
                    ['push', 'origin', 'master'],
            ]:
                Release.run_subprocess(
                    ['git'] + argv,
                    cwd=project_root,
                )
        except:  # TODO: exc types
            raise NotImplementedError

        return Release.run_subprocess(
            ['git', 'rev-parse', 'master'],
            cwd=project_root,
        ).stdout.strip()  # Get rid of the newline character at the end

    @staticmethod
    def git_merge_master_to_develop(project_root: str):  # TODO: accept commit hash?
        """Merge the master branch back into develop after the release is performed."""
        for argv in [
                ['fetch', '-p'],
                ['checkout', 'develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/develop'],
                ['merge', '--no-ff', '--no-edit', 'origin/master'],
                ['push', 'origin', 'develop'],
        ]:
            Release.run_subprocess(
                ['git'] + argv,
                cwd=project_root,
            )

    def get_project_root(self, project_name: str) -> str:
        """Get the root of the project's Git repo locally."""
        return self.config['REPOS_ROOT'] + project_name

    def get_jira_latest_version(
            self,
            project_key: str,
    ) -> 'jira.resources.Version':
        """Get the latest version resource from JIRA.

        Assumes all existing versions are released."""
        try:
            return self.jira_client.project_versions(project_key)[-1]
        except (JIRAError, IndexError) as exc:
            self.log.exception(
                'Unable to get the latest JIRA version resource for project_key=%s',
                project_key,
            )
            raise exc

    def get_jira_release_type(self, project_key: str):
        """Get the highest Release Type of all closed issues without a Fix Version."""
        try:
            for release_type in ['Major', 'Minor', 'Patch']:
                if len(
                        self.jira_client.search_issues(
                            jql_str=(
                                'project = {project_key} '
                                'AND status = "closed" '
                                'AND fixVersion = EMPTY '
                                'AND resolution in ("Fixed", "Done") '
                                'AND "Release Type" = "{release_type}" '
                            ).format(
                                project_key=project_key.upper(),
                                release_type=release_type,
                            ),
                        )
                ) > 0:
                    return release_type
        except JIRAError as exc:
            self.log.exception(
                'Unknown JIRA error occurred when trying to determine release type for project_key=%s',
                project_key,
            )
            raise exc

        raise NoJIRAIssuesFoundError(
            'Could not find any closed issues without a fixVersion in project_key=%s' % project_key
        )

    def get_jira_release_notes(self, version: 'jira.resources.Version') -> str:
        """Produce release notes for a JIRA project version."""
        env = Environment(
            loader=FileSystemLoader(self.config['TEMPLATE_DIR']),
            lstrip_blocks=True,
            trim_blocks=True,
        )
        template = env.get_template('release_notes.html')

        project_name = self.jira_client.project(version.projectId).name
        try:
            return template.render({
                'project_name': project_name,
                'version_number': version.name,
                'issues': self.jira_client.search_issues(
                    jql_str=(
                        'project = {project_name} '
                        'AND fixVersion = "{version_name}" '
                        'ORDER BY issuetype ASC, updated DESC'
                    ).format(
                        project_name=project_name,
                        version_name=version.name,
                    ),
                ),
            })
        except JIRAError as exc:
            logger.exception(
                'Could not retrieve issues for %s v%s.',
                project_name,
                version.name,
            )
            raise exc

    def set_jira_fix_version(self, project_key: str, new_version: str):
        """Set the fixVersion on all of the closed tickets without one."""
        # TODO: exceptions

        for issue in self.jira_client.search_issues(
                jql_str=(
                    'project = "{}" '
                    'AND status = "closed" '
                    'AND resolution in ("Fixed", "Done") '
                    'AND fixVersion = EMPTY'
                ).format(
                    project_key.upper(),
                ),
        ):
            self.jira_client.transition_issue(issue, 'Reopen Issue')

            issue.update(
                fixVersions=[
                    {
                        # Add new fix version to the existing versions
                        'add': {'name': new_version}
                    }
                ]
            )

            self.jira_client.transition_issue(issue, 'Close Issue')

    def get_jira_client(self) -> JIRA:
        """Get an instance of the JIRA client using the plugins configuration for authentication."""
        jira_url = self.config['JIRA_URL']

        try:
            client = JIRA(
                server=jira_url,
                basic_auth=(
                    self.config['JIRA_USER'],
                    self.config['JIRA_PASS'],
                ),
            )
            self.log.info('Initialized JIRA client at URL %s', jira_url)
            return client
        except JIRAError as exc:
            self.log.exception('Unable to initialize JIRA client at URL=%s', jira_url)
            raise exc

    def get_github_client(self) -> Github:
        """Get an instance of the PyGitHub client using the plugins configuration for authentication."""
        return Github(self.config['GITHUB_TOKEN'])

    @staticmethod
    def bump_version(version: str, release_type: str) -> str:
        """Perform a version bump in accordance with semver.org."""
        MAJOR = 0
        MINOR = 1
        PATCH = 2

        # TODO: exceptions
        release_type = vars()[release_type.upper()]
        version_array = [int(x) for x in version.split('.', 2)]
        version_array[release_type] += 1
        version_array[release_type+1:] = [0] * (PATCH - release_type)

        return '.'.join(str(version) for version in version_array)

    @staticmethod
    def run_subprocess(args: list, cwd: str=None):
        """Run the local command described by `args` with some defaults applied."""
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine out/err into stdout; stderr will be None
            universal_newlines=True,
            check=True,
            cwd=cwd,
        )
