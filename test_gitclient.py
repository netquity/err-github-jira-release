import os
import re
from shutil import rmtree

import pytest
from gitclient import RepoManager
from pytest_shutil.workspace import Workspace
from git import Repo

REPOS_ROOT = os.path.join(os.path.dirname(__file__), 'testing/repos/')
GH_TOKEN = 'fake token'
PROJECT_NAME = 'netquity/err-github-jira-release'
PROJECT_NAMES = [PROJECT_NAME, ]


@pytest.yield_fixture
def git_repo(request):  # pylint:disable=unused-argument
    """ Function-scoped fixture to create a new git repo in a temporary workspace.

        Attributes
        ----------
        uri (str) :  Repository URI
        api (`git.Repo`) :  Git Repo object for this repository
        .. also inherits all attributes from the `workspace` fixture

    """
    workspace_path = os.path.join(REPOS_ROOT, PROJECT_NAME)
    with GitRepo(workspace_path) as repo:
        yield repo


class GitRepo(Workspace):  # pylint:disable=too-few-public-methods
    """
    Creates an empty Git repository in a temporary workspace.
    Cleans up on exit.
    Attributes
    ----------
    uri : `str`
        repository base uri
    api : `git.Repo` handle to the repository
    """
    def __init__(self, workspace):
        super(GitRepo, self).__init__(workspace)
        self.api = Repo.init(self.workspace)
        self.uri = "file://%s" % self.workspace


@pytest.fixture
def repomanager(request, git_repo):  # pylint:disable=redefined-outer-name
    """A RepoManager fixture to use in tests"""
    # Add a single commit
    path = git_repo.workspace
    file = path / 'hello.txt'
    file.write_text('hello world!')
    git_repo.run('git add hello.txt')
    git_repo.api.index.commit("Initial commit")

    repomanager = RepoManager({
        'REPOS_ROOT': os.path.dirname(git_repo.workspace.parent),
        'GITHUB_TOKEN': GH_TOKEN,
        'PROJECT_NAMES': [PROJECT_NAME, ],
    })

    def delete_backups():
        backups_dir = repomanager._get_backups_path()
        if os.path.exists(backups_dir):
            rmtree(backups_dir)

    request.addfinalizer(delete_backups)
    return repomanager


def test_get_project_root(repomanager):  # pylint:disable=redefined-outer-name
    """Check the formatting of the project root string"""
    assert (
        repomanager._get_project_root(PROJECT_NAME)
        == os.path.join(repomanager.repos_root, PROJECT_NAME)
    )


def test_get_ref(git_repo, repomanager):  # pylint:disable=redefined-outer-name
    assert git_repo.api.git.reflog('--format=%H', '-1') == repomanager.get_ref(
        PROJECT_NAME
    )


def test_git_context_manager(repomanager):  # pylint:disable=redefined-outer-name
    """Check that the context manager passes through commands and rolls back on error"""
    with repomanager._gcmd(PROJECT_NAME) as gcm:  # pylint:disable=protected-access
        assert 'working tree clean' in gcm(['status']).stdout


def test_backup_repo(repomanager):  # pylint:disable=redefined-outer-name
    """Repo backups go to the right place without modification"""
    repomanager._get_project_root(PROJECT_NAME)
    dst = repomanager._backup_repo(PROJECT_NAME)

    # confirm the expected location
    assert re.match(f'{REPOS_ROOT}.backups/{PROJECT_NAME}/[a-f0-9]{{7}}', dst)
    # FIXME: would like to be able to assert that after the backup, the repo and backup dir are identical
    # assert not filecmp.dircmp(src, dst).diff_files


def test_get_backups_path(repomanager):  # pylint:disable=redefined-outer-name
    """Check that the backups are goign to the right place"""
    assert re.match(f'{REPOS_ROOT}.backups', repomanager._get_backups_path())
    assert re.match(f'{REPOS_ROOT}.backups/{PROJECT_NAME}', repomanager._get_backups_path(PROJECT_NAME))


def test_restore_repo(git_repo, repomanager):  # pylint:disable=redefined-outer-name
    """Check that repo backups are properly restored"""
    ref = repomanager.get_ref(PROJECT_NAME)[:7]
    backup_path = os.path.join(repomanager._get_backups_path(PROJECT_NAME), ref)
    repomanager._get_project_root(PROJECT_NAME)

    repomanager._backup_repo(PROJECT_NAME)

    path = git_repo.workspace
    file = path / 'goodbye.txt'
    file.write_text('goodbye world!')
    git_repo.run('git add goodbye.txt')
    git_repo.api.index.commit("Final commit")

    assert ref != repomanager.get_ref(PROJECT_NAME)[:7]  # ref was updated with the last commit

    repomanager._restore_repo(PROJECT_NAME, backup_path)

    assert ref == repomanager.get_ref(PROJECT_NAME)[:7]  # original ref restored after _restore_repo

    assert os.path.exists(backup_path)  # the backup should still be available after restoring
    assert not os.path.exists(backup_path + '.swap')  # delete the backup swap
