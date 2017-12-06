# coding: utf-8
import datetime
import logging

import utils

logger = logging.getLogger(__file__)

from jinja2 import Environment, FileSystemLoader

try:
    from jira import JIRA, JIRAError
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")


class JIRAVersionError(Exception):
    """Could not find the given JIRA version resource."""


class NoJIRAIssuesFoundError(Exception):
    """No JIRA issues match the given search parameters."""


class JiraClient:
    def __init__(self, config: dict):
        self.project_key = config['PROJECT_KEY']
        self.template_dir = config['TEMPLATE_DIR']
        try:
            self.api = JIRA(
                server=config['URL'],
                basic_auth=(
                    config['USER'],
                    config['PASS'],
                ),
            )
            logger.info('Initialized JIRA client at URL %s', config['URL'])
        except JIRAError as exc:
            logger.exception('Unable to initialize JIRA client at URL=%s', config['URL'])
            raise exc

    def get_project_name(self):
        return self.api.project(self.project_key).name

    def delete_version(self, version: 'jira.resources.Version', failed_command: str='JIRA'):
        """Delete a JIRA version.

        Used to undo created versions when subsequent operations fail."""
        try:
            version.delete()  # Remove version from issues it's attached to
        except JIRAError:
            exc_message = (
                'Unable to complete JIRA request for project_key={} and unable to clean up new version={}'.format(
                    self.project_key,
                    version.name,
                )
            )
            return exc_message

        return 'Unable to complete %s operation for project_key=%s. JIRA version deleted.' % (
            failed_command,
            self.project_key,
        )

    def get_latest_version(self) -> 'jira.resources.Version':
        """Get the latest version resource from JIRA.

        Assumes all existing versions are released and ordered by descending date."""
        try:
            return self.api.project_versions(self.project_key)[-1]
        except (JIRAError, IndexError) as exc:
            logger.exception(
                'Unable to get the latest JIRA version resource for project_key=%s',
                self.project_key,
            )
            raise exc

    def get_release_notes(self, version: 'jira.resources.Version') -> str:
        """Produce release notes for a JIRA project version."""
        template = Environment(
            loader=FileSystemLoader(self.template_dir),
            lstrip_blocks=True,
            trim_blocks=True,
        ).get_template('release_notes.html')

        project_name = self.api.project(version.projectId).name
        try:
            return template.render({
                'project_name': project_name,
                'version_number': version.name,
                'issues': self.api.search_issues(
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

    def get_release_type(self, is_hotfix: bool=False) -> str:
        """Get the highest Release Type of all closed issues without a Fix Version."""
        if is_hotfix:
            return 'Hotfix'
        try:
            for release_type in ['Major', 'Minor', 'Patch']:
                if len(
                        self.api.search_issues(
                            jql_str=(
                                'project = {project_key} '
                                'AND status = "closed" '
                                'AND fixVersion = EMPTY '
                                'AND resolution in ("Fixed", "Done") '
                                'AND "Release Type" = "{release_type}" '
                            ).format(
                                project_key=self.project_key.upper(),
                                release_type=release_type,
                            ),
                        )
                ) > 0:  # Since we go in order from highest to smallest, pick the first one found
                    return release_type
        except JIRAError as exc:
            logger.exception(
                'Unknown JIRA error occurred when trying to determine release type for project_key=%s',
                self.project_key,
            )
            raise exc

        raise NoJIRAIssuesFoundError(
            'Could not find any closed issues without a fixVersion in project_key=%s' % self.project_key
        )

    def get_release_url(self, version_id: int) -> str:
        return '{jira_url}/projects/{project_key}/versions/{version_id}/tab/release-report-done'.format(
            jira_url=self.api.client_info(),
            project_key=self.project_key,
            version_id=version_id,
        )

    def set_fix_version(self, new_version: str):
        """Set the fixVersion on all of the closed tickets without one."""
        # TODO: exceptions

        for issue in self.api.search_issues(
                jql_str=(
                    'project = "{}" '
                    'AND status = "closed" '
                    'AND resolution in ("Fixed", "Done") '
                    'AND fixVersion = EMPTY'
                ).format(
                    self.project_key.upper(),
                ),
        ):
            self.api.transition_issue(issue, 'Reopen Issue')

            issue.update(
                fixVersions=[
                    {
                        # Add new fix version to the existing versions
                        'add': {'name': new_version}
                    }
                ]
            )

            self.api.transition_issue(issue, 'Close Issue')

    def create_version(self, release_type: str) -> 'jira.resources.Version':
        return self.api.create_version(
            utils.bump_version(
                self.get_latest_version().name,
                release_type,
            ),
            project=self.project_key,
            released=True,
            releaseDate=datetime.datetime.now().date().isoformat(),
        )
