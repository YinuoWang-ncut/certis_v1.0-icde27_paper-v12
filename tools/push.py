#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Push the reviewed tree and the v1.0-icde27 tag to GitHub, the ordinary way.

Run this from a shell that has normal internet access and a logged-in git
credential helper:

    python tools/push.py --check      # show what would be sent, change nothing
    python tools/push.py              # push main and the tag, then verify

Why a script instead of three commands: the remote already holds one unrelated
commit (the v12 archive, uploaded through the web UI), so the branch update has
to be forced; and the tag is annotated and has to be re-pointed whenever a fix
lands. Both are easy to do in the wrong order and hard to notice afterwards, so
this checks the result against the GitHub API instead of trusting git's exit
code.

A .bat or .ps1 helper would be the obvious wrapper, but the repository path
contains non-ASCII characters, which those two formats mangle on Windows.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REMOTE = 'origin'
BRANCH = 'main'
TAG = 'v1.0-icde27'


def git(*args, check=True, timeout=180, retries=0, prompt=False):
    p = subprocess.run([GIT] + list(args), cwd=ROOT, capture_output=True,
                       text=True, errors='replace', timeout=timeout,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    if check and p.returncode != 0:
        sys.exit('git %s failed:\n%s' % (' '.join(args), (p.stderr or p.stdout).strip()))
    return p


def find_git():
    for c in glob.glob(r'C:/Users/Admin/.workbuddy/binaries/PortableGit/**/git.exe',
                       recursive=True):
        return c
    return 'git'


GIT = find_git()


def in_repo():
    p = git('rev-parse', '--git-dir', check=False)
    return p.returncode == 0


def remote_slug():
    """owner/repo from the origin URL, whatever form it was written in."""
    p = git('config', '--get', 'remote.%s.url' % REMOTE, check=False)
    url = (p.stdout or '').strip() if p.returncode == 0 else ''
    if not url:
        print('no remote named %r: run  git remote add origin <url>  once, then retry' % REMOTE)
        return None, None
    slug = url.rstrip('/')
    for pre in ('https://github.com/', 'http://github.com/', 'git@github.com:'):
        if slug.startswith(pre):
            slug = slug[len(pre):]
    if slug.endswith('.git'):
        slug = slug[:-4]
    return slug, url


def api(path):
    req = urllib.request.Request('https://api.github.com/' + path,
                                 headers={'User-Agent': 'certis-push-check',
                                          'Accept': 'application/vnd.github+json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def report():
    slug, url = remote_slug()
    if slug is None:
        return None
    print('remote      : %s  (%s)' % (url, slug))
    print('local head  : %s' % git('rev-parse', '--short', 'HEAD').stdout.strip())
    print('local tag   : %s -> %s, tree %s' % (
        TAG,
        git('rev-parse', '--short', TAG + '^{commit}').stdout.strip(),
        git('rev-parse', TAG + '^{tree}').stdout.strip()))
    n = len(git('ls-tree', '-r', TAG, '--name-only').stdout.splitlines())
    print('tree files  : %d' % n)
    dirty = [l for l in git('status', '--porcelain').stdout.splitlines() if l.strip()]
    if dirty:
        print('working tree: %d uncommitted entries -- they are NOT pushed, only the '
              'tree of %s is' % (len(dirty), TAG))
    try:
        info = api('repos/' + slug)
        print('remote repo : %d KB, default branch %r, pushed %s'
              % (info.get('size', 0), info.get('default_branch'),
                 (info.get('pushed_at') or '')[:19]))
        try:
            tags = api('repos/%s/git/refs/tags' % slug)
            print('remote tags : %s' % ([t['ref'] for t in tags] or 'none'))
        except Exception:
            print('remote tags : none')       # GitHub answers 404 when no tag exists
    except Exception as e:
        print('remote check: %s' % e)
    return slug


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='report, change nothing')
    ap.add_argument('--no-prompt', action='store_true',
                    help='never ask for credentials (CI); default lets the helper log in')
    a = ap.parse_args()

    if not in_repo():
        # Running from an unpacked archive is a normal mistake, not an error:
        # say so and stop, because there is nothing here to push.
        print('%s is not a git repository.' % ROOT)
        print('This tool has to run inside the working tree, not in the unpacked '
              'artifact.  If you only have the archive, the tree it came from is '
              'already the reviewed one.')
        return 0

    slug = report()
    if slug is None:
        return 1
    if a.check:
        return 0

    print('\n--- push %s ---' % BRANCH)
    # The remote history is unrelated (one web-upload commit), so this has to
    # be forced.  The replaced commit stays in GitHub's reflog for 90 days.
    p = git('push', '--force', REMOTE, 'refs/heads/%s:refs/heads/%s' % (BRANCH, BRANCH),
            check=False, retries=5, prompt=not a.no_prompt)
    print((p.stdout + p.stderr).strip() or '(no output)')
    if p.returncode != 0:
        blob = (p.stderr or '') + (p.stdout or '')
        print('\nBranch push failed.  The two causes are distinguishable:')
        if 'password' in blob.lower() or 'authentication' in blob.lower() or '403' in blob:
            print('  AUTH: no usable credential.  Log in once, then re-run:')
            print('        git credential-manager github login')
            print('    or give tools/push_api.py a fine-grained token with')
            print('    Contents: read and write and let it push through the API.')
        elif '502' in blob or 'tunnel' in blob.lower():
            print('  NETWORK: github.com is blocked by a proxy here.  Use')
            print('           python tools/push_api.py, which talks to api.github.com.')
        else:
            print('  Unrecognised failure; the message above is the whole of it.')
        return 1

    print('\n--- push tag %s ---' % TAG)
    p = git('push', '--force', REMOTE, 'refs/tags/%s:refs/tags/%s' % (TAG, TAG),
            check=False, retries=5, prompt=not a.no_prompt)
    print((p.stdout + p.stderr).strip() or '(no output)')
    if p.returncode != 0:
        return 1

    print('\n--- verify against the API ---')
    try:
        tags = api('repos/%s/git/refs/tags' % slug)
        have = [t['ref'] for t in tags]
        print('remote tags:', have)
        if 'refs/tags/' + TAG not in have:
            print('FAIL: tag %s is not on the remote' % TAG)
            return 1
        ref = [t for t in tags if t['ref'] == 'refs/tags/' + TAG][0]['object']['sha']
        local_tree = git('rev-parse', TAG + '^{tree}').stdout.strip()
        remote_tree = api('repos/%s/git/trees/%s' % (slug, TAG))['sha']
        print('local tree :', local_tree)
        print('remote tree:', remote_tree)
        if local_tree != remote_tree:
            print('FAIL: the trees differ -- the remote does not hold the reviewed bytes')
            return 1
        print('OK: the remote tag holds exactly the tree that was reviewed')
        print('    %s/tree/%s' % ('https://github.com/' + slug, TAG))
    except Exception as e:
        print('could not verify (%s); check %s/tree/%s by hand'
              % (e, 'https://github.com/' + slug, TAG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
