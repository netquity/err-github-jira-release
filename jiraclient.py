# coding: utf-8
"""
This module provides a wrapper for the Jira REST API. Only a few of the API methods are supported, as the main goal is
to provide release-management capabilities and nothing more.
"""
import datetime
import logging
import os

from typing import Optional, List

from jinja2 import Environment, FileSystemLoader

from helpers import Stages

from gitclient import MergeLog


logger = logging.getLogger(os.path.basename(__file__))


try:
    from jira import JIRA, JIRAError
    from jira.resources import Version
except ImportError:
    logger.error("Please install 'jira' and 'pygithub' Python packages")


class JiraClientError(Exception):
    """Module-level exception class"""


class JIRAVersionError(JiraClientError):
    """Could not find the given JIRA version resource."""


class NoJIRAIssuesFoundError(JiraClientError):
    """No JIRA issues match the given search parameters."""


class IssueMergesCountMismatchError(JiraClientError):
    """The number pending Jira issues does not match the number of merged PRs."""


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

    def get_latest_version(self, project_key: str) -> Version:
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
            sha: str,
            final_version: str,
            pre_version: str = None,
    ) -> str:
        """Get a project's next version number, including merged, but yet unreleased, tickets

        :param stage: the release stage to transition into (seal, send, sign)
        :param sha: short commit hash to append to metadata segment, usually origin/develop's HEAD
        :param final_version: the latest final version name; the version to be upgraded (eg. 1.0.0)
        :param pre_version: the version containing information about the current release cycle: prerelease version, rc
                            count, stage (eg. 1.0.1-rc.5+sealed) it's possible that such a version does not yet exist,
                            as when initiating the release sequence for the first time
        """
        from helpers import bump_version
        return bump_version(
            release_type=self.get_release_type(project_key),
            stage=stage.verb,
            sha=sha,
            final_version=final_version[1:],  # dropping the leading `v`  TODO: do it better
            pre_version=pre_version[1:] if pre_version else None,
        )

    def get_release_notes(self, version: Version, merge_logs: List[MergeLog]) -> str:
        """Produce release notes for a JIRA project version."""
        template = Environment(
            loader=FileSystemLoader(self.template_dir),
            lstrip_blocks=True,
            trim_blocks=True,
        ).get_template('release_notes.html')

        project_name = self.api.project(version.projectId).name
        try:
            issues = self.api.search_issues(
                jql_str=(
                    'project = {project_name} '
                    'AND fixVersion = "{version_name}" '
                    'ORDER BY issuetype ASC, updated DESC'
                ).format(
                    project_name=project_name,
                    version_name=version.name,
                ),
            )
            logger.debug(
                '%s: %s issues found for release notes, %s merge logs found in git',
                project_name,
                len(issues),
                len(merge_logs),
            )

            issues.sort(key=lambda issue: issue.key)
            if len(issues) != len(merge_logs):
                logger.error(
                    '%s: Jira issue count (%s) does not match merge count (%s) for %s',
                    project_name,
                    len(issues),
                    len(merge_logs),
                    version.name,
                )
                raise IssueMergesCountMismatchError(
                    '%s got %s Jira issues but %s merged PRs.' %
                    project_name,
                    len(issues),
                    len(merge_logs),
                )
            for issue in issues:
                issue.sha = next(merge.sha for merge in merge_logs if merge.key == issue.key)

            return template.render({
                'project_name': project_name,
                'version_number': version.name,
                'issues': issues,
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

    def set_fix_version(self, project_key: str, new_version: str, is_hotfix: bool = False) -> None:
        """Set the fixVersion on all of the closed tickets without one."""
        # TODO: exceptions
        issues = self.api.search_issues(
            # For non-hotfix releases the release type isn't part of the search criteria since it's a mixture
            jql_str=self.get_issue_search_string(project_key) + (
                'AND "Release Type" = "Hotfix"' if is_hotfix else ''
            )
        )
        logger.info('%s: starting setting fixVersion %s on %s issues', project_key.upper(), new_version, len(issues))
        for issue in issues:
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
            logger.info('%s: set fixVersion %s on %s', project_key.upper(), new_version, issue.key)
        logger.info('%s: set fixVersion %s on %s issues', project_key.upper(), new_version, len(issues))

    def create_version(self, project_key: str, new_version: str, released: bool = False) -> Version:
        """Create a Jira version, applying the appropriate version bump"""
        version = self.api.create_version(
            new_version,
            project=project_key.upper(),
            released=released,
            releaseDate=datetime.datetime.now().date().isoformat(),
        )
        logger.info('%s: created new Jira version %s (released=%s)', project_key.upper(), new_version, released)
        return version

    def change_version_name(self, version: Version, new_name: str) -> Version:
        """Change the Jira version's name"""
        original_name = version.name
        version.update(name=new_name)
        logger.info('Changed Jira version %s to %s', original_name, new_name)
        return version

    # Doesn't work, probably need to pull the logo from GitHub
    # def get_project_avatar(self, project_key: str) -> str:
    #     return self.api.project(project_key).raw['avatarUrls']['48x48']

    @staticmethod
    def delete_version(project_key: str, version: Version, failed_command: str = 'JIRA') -> Optional[str]:
        """Delete a JIRA version.

        Used to undo created versions when subsequent operations fail."""
        try:
            version.delete()  # Remove version from issues it's attached to
            logger.info('%s: deleted Jira version %s', project_key.upper(), version.name)
            return
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

    @staticmethod
    def get_issue_search_string(project_key: str) -> str:
        """Search for issues in transition since the last release"""
        # TODO: maybe we should search by some other field that unites all projects in the stack
        return 'project = %s ' % project_key.upper() + (
            'AND status = "closed" '
            'AND fixVersion = EMPTY '
            'AND resolution in ("Fixed", "Done") '
        )
