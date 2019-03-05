# coding: utf-8
"""
This module provides a wrapper for the Jira REST API. Only a few of the API methods are supported, as the main goal is
to provide release-management capabilities and nothing more.
"""
import datetime
import logging

from jinja2 import Environment, FileSystemLoader

from helpers import Stages

logger = logging.getLogger(__file__)


try:
    from jira import JIRA, JIRAError, resources
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")


class JIRAVersionError(Exception):
    """Could not find the given JIRA version resource."""


class NoJIRAIssuesFoundError(Exception):
    """No JIRA issues match the given search parameters."""


class JiraClient:
    """Limited wrapper for the Jira REST API"""
    def __init__(self, config: dict):
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

    def get_project_name(self, project_key: str) -> str:
        """Get the Jira project name for the given project key"""
        return self.api.project(project_key.upper()).name

    def get_latest_version(self, project_key: str) -> resources.Version:
        """Get the latest version resource from JIRA.

        Assumes all existing versions are released and ordered by descending date."""
        try:
            return self.api.project_versions(project_key.upper())[-1]
        except (JIRAError, IndexError) as exc:
            logger.exception(
                'Unable to get the latest JIRA version resource for project_key=%s',
                project_key.upper(),
            )
            raise exc

    def get_pending_version_name(
            self,
            project_key: str,
            stage: Stages,
            final_version: str,
            pre_version: str = None
    ) -> str:
        """Get a project's next version number, including merged, but yet unreleased, tickets"""
        from helpers import bump_version
        return bump_version(
            release_type=self.get_release_type(project_key),
            stage=stage.verb,
            final_version=final_version[1:],  # dropping the leading `v`
            pre_version=pre_version,
        )

    def get_release_notes(self, version: resources.Version) -> str:
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

    def get_release_type(self, project_key: str) -> str:
        """Get the highest Release Type of all closed issues without a Fix Version."""
        search_string = self.get_issue_search_string(project_key) + 'AND "Release Type" = "{release_type}" '
        try:
            for release_type in ['Hotfix', 'Major', 'Minor', 'Patch']:
                if (
                        self.api.search_issues(
                            jql_str=search_string.format(release_type=release_type),
                        )
                ):  # Since we go in order from highest to smallest, pick the first one found
                    return release_type
        except JIRAError as exc:
            logger.exception(
                'Unknown JIRA error occurred when trying to determine release type for project_key=%s',
                project_key.upper(),
            )
            raise exc

        raise NoJIRAIssuesFoundError(
            'Could not find any closed issues without a fixVersion in project_key=%s' % project_key.upper()
        )

    def get_latest_issues_url(self, project_key: str) -> str:
        """Get the Jira issue URL showing all closed issues with a fixVersion"""
        from urllib import parse
        return '{jira_url}/issues/?jql={query}'.format(
            jira_url=self.api.client_info(),
            query=parse.quote_plus(
                f'project = {project_key.upper()} AND status = "closed" '
                'AND fixVersion = EMPTY '
                'AND resolution in ("Fixed", "Done") '
                'ORDER BY created DESC'
            ),
        )

    def get_release_url(self, project_key: str, version_id: int) -> str:
        """Get the URL of the Jira version object"""
        return '{jira_url}/projects/{project_key}/versions/{version_id}/tab/release-report-done'.format(
            jira_url=self.api.client_info(),
            project_key=project_key.upper(),
            version_id=version_id,
        )

    def set_fix_version(self, project_key: str, new_version: str, is_hotfix: bool = False):
        """Set the fixVersion on all of the closed tickets without one."""
        # TODO: exceptions
        for issue in self.api.search_issues(
                # For non-hotfix releases the release type isn't part of the search criteria since it's a mixture
                jql_str=self.get_issue_search_string(project_key) + (
                    'AND "Release Type" = "Hotfix"' if is_hotfix else ''
                )
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

    def create_version(self, project_key: str, new_version: str) -> resources.Version:
        """Create a Jira version, applying the appropriate version bump"""
        return self.api.create_version(
            new_version,
            project=project_key.upper(),
            released=True,
            releaseDate=datetime.datetime.now().date().isoformat(),
        )

    @classmethod
    def delete_version(cls, project_key: str, version: resources.Version, failed_command: str = 'JIRA'):
        """Delete a JIRA version.

        Used to undo created versions when subsequent operations fail."""
        try:
            version.delete()  # Remove version from issues it's attached to
        except JIRAError:
            exc_message = (
                'Unable to complete JIRA request for project_key={} and unable to clean up new version={}'.format(
                    project_key.upper(),
                    version.name,
                )
            )
            return exc_message

        return 'Unable to complete %s operation for project_key=%s. JIRA version deleted.' % (
            failed_command,
            project_key.upper(),
        )

    @classmethod
    def get_issue_search_string(cls, project_key: str) -> str:
        """Search for issues in transition since the last release"""
        # TODO: maybe we should search by some other field that unites all projects in the stack
        return 'project = %s ' % project_key.upper() + (
            'AND status = "closed" '
            'AND fixVersion = EMPTY '
            'AND resolution in ("Fixed", "Done") '
        )
