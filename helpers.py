import os
import shutil
import subprocess
import logging


def update_changelog_file(
        changelog_path: str,
        release_notes: str,
        log: logging.Logger
):
    """Prepend the given release notes to CHANGELOG.md."""
    # TODO: exceptions
    try:
        with open(changelog_path, 'r') as changelog:
            original_contents = changelog.read()
        with open(changelog_path, 'w') as changelog:
            changelog.write(release_notes + "\n" + original_contents)
    except OSError as exc:
        log.exception('An unknown error occurred while updating the changelog file.')
        raise exc


def run_subprocess(args: list, cwd: str = None) -> subprocess.CompletedProcess:
    """Run the local command described by `args` with some defaults applied."""
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine out/err into stdout; stderr will be None
        universal_newlines=True,
        check=True,
        cwd=cwd,
    )


def bump_version(version: str, release_type: str) -> str:
    """Perform a version bump in accordance with semver.org."""
    if release_type == 'Hotfix':  # Don't bump the version number, just append the suffix
        return str(version) + '-Hotfix'
    MAJOR = 0
    MINOR = 1
    PATCH = 2

    # TODO: exceptions
    release_type = vars()[release_type.upper()]
    version_array = [int(x) for x in version.split('.', 2)]
    version_array[release_type] += 1
    version_array[release_type + 1:] = [0] * (PATCH - release_type)

    return '.'.join(str(version) for version in version_array)


def copytree(src: str, dst_parent: str, dst: str) -> str:
    """A simple wrapper for `shutil.copytree` which makes sure required parent directory exists

    :param src: the dir to be copied
    :param dst_parent: the parent location to copy the src to
    :param dst: the name of the folder to hold a copy of the src
    """
    if not os.path.exists(dst_parent):
        os.makedirs(dst_parent)  # creates intermediate dirs

    return shutil.copytree(src, f'{dst_parent}/{dst}')
