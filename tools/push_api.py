#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Push the reviewed tree and tag through api.github.com instead of git.

Use this when `git push` cannot reach GitHub: git's smart-HTTP goes to
github.com, and a proxy that blocks that host returns

    fatal: unable to access 'https://github.com/...': CONNECT tunnel failed,
    response 502

while leaving api.github.com reachable.  The Git Data API performs the same
three operations git would -- write blobs, build a tree, move the ref -- so the
result is a real commit on the branch, not an upload of an archive.

The check that matters is at the end and it runs before any ref is touched: the
tree GitHub reports for the commit we just created has to equal the tree of the
local annotated tag, byte for byte.  If it does not, the script stops and leaves
the branch where it was, because a remote that holds "roughly" the reviewed tree
is worse than one that holds the old one.

    export GITHUB_TOKEN=...          # fine-grained PAT, Contents: read and write
    python tools/push_api.py --plan  # what would be sent, no writes
    python tools/push_api.py         # create the commit, move the branch, tag it

The token is read from the environment or asked for with getpass; it is never
written to disk, never put in a command line, and never printed.
"""
import argparse
import base64
import getpass
import glob
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TAG = 'v1.0-icde27'
API = 'https://api.github.com'
BRANCH_FALLBACK = 'main'


def find_git():
    for c in glob.glob(r'C:/Users/Admin/.workbuddy/binaries/PortableGit/**/git.exe',
                       recursive=True):
        return c
    return 'git'


GIT = find_git()


def git(*args, binary=False):
    p = subprocess.run([GIT] + list(args), cwd=ROOT, capture_output=True)
    if p.returncode != 0:
        sys.exit('git %s failed: %s' % (' '.join(args),
                                        p.stderr.decode('utf-8', 'replace').strip()))
    return p.stdout if binary else p.stdout.decode('utf-8', 'replace')


def local_tree():
    """(tree sha, [(path, mode, sha), ...]) for the annotated tag."""
    tree = git('rev-parse', TAG + '^{tree}').strip()
    out = git('ls-tree', '-r', '-z', TAG, binary=True)
    entries = []
    for rec in out.split(b'\0'):
        if not rec:
            continue
        meta, path = rec.split(b'\t', 1)
        mode, typ, sha = meta.decode().split()
        if typ != 'blob':
            continue
        entries.append((path.decode('utf-8'), mode, sha))
    return tree, entries


class Client(object):
    def __init__(self, token, slug):
        self.token = token
        self.slug = slug

    def call(self, method, path, payload=None, raw=False):
        url = path if path.startswith('http') else API + path
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'certis-push-api',
            'Content-Type': 'application/json',
        }
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        _unused = {
        }
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                return body if raw else (json.loads(body) if body else None)
        except urllib.error.HTTPError as e:
            detail = e.read().decode('utf-8', 'replace')[:400]
            raise RuntimeError('%s %s -> %d %s\n%s'
                               % (method, path, e.code, e.reason, detail))


def read_bytes(path):
    with open(os.path.join(ROOT, path), 'rb') as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', action='store_true', help='report, write nothing')
    ap.add_argument('--token', default=None,
                    help='prefer the GITHUB_TOKEN environment variable')
    ap.add_argument('--slug', default='YinuoWang-ncut/certis_v1.0-icde27_paper-v12')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()

    tree, entries = local_tree()
    total = sum(len(read_bytes(p)) for p, _, _ in entries)
    print('local tag   : %s' % TAG)
    print('local tree  : %s' % tree)
    print('files       : %d  (%.2f MB)' % (len(entries), total / 1e6))

    token = a.token or os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token and not a.plan:
        token = getpass.getpass('GitHub token (Contents: read and write, not echoed): ')
    if not token and not a.plan:
        sys.exit('no token')

    # --plan can run unauthenticated: every call it makes is a read, and
    # api.github.com serves those anonymously within its rate limit.
    cli = Client(token or '', a.slug)

    try:
        repo = cli.call('GET', '/repos/' + a.slug)
        branch = repo.get('default_branch') or BRANCH_FALLBACK
        print('remote repo : %s, default branch %r, %d KB'
              % (repo['full_name'], branch, repo.get('size', 0)))
    except Exception as e:
        print('cannot read the repository: %s' % e)
        print('(api.github.com is reachable from here; a 401 means the token is wrong)')
        return 1

    try:
        head = cli.call('GET', '/repos/%s/git/ref/heads/%s' % (a.slug, branch))['object']['sha']
        print('remote head : %s' % head)
    except Exception:
        head = None
        print('remote head : (branch has no commits yet)')

    # does the remote already hold this tree?
    if head:
        try:
            rt = cli.call('GET', '/repos/%s/git/commits/%s' % (a.slug, head))['tree']['sha']
            if rt == tree:
                print('\nremote head already carries tree %s -- nothing to do' % tree)
                return 0
        except Exception:
            pass

    if a.plan:
        print('\nplan: %d blobs -> 1 tree -> 1 commit on %r -> tag %s'
              % (len(entries), branch, TAG))
        return 0

    print('\n--- uploading %d blobs ---' % len(entries))

    def put(item):
        path, mode, sha = item
        body = cli.call('POST', '/repos/%s/git/blobs' % a.slug,
                        {'content': base64.b64encode(read_bytes(path)).decode(),
                         'encoding': 'base64'})
        if body['sha'] != sha:
            raise RuntimeError('blob %s: local %s, remote %s' % (path, sha, body['sha']))
        return path, mode, body['sha']

    blobs = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(put, entries), 1):
            blobs.append(res)
            if i % 50 == 0 or i == len(entries):
                print('  %d/%d' % (i, len(entries)))

    print('--- building the tree ---')
    new_tree = cli.call('POST', '/repos/%s/git/trees' % a.slug, {
        'tree': [{'path': p, 'mode': m if m in ('100644', '100755', '120000') else '100644',
                  'type': 'blob', 'sha': s} for p, m, s in blobs]})
    if new_tree['sha'] != tree:
        print('STOP: the tree GitHub built (%s) is not the local one (%s).'
              % (new_tree['sha'], tree))
        print('Nothing was moved; the branch still points at %s.' % head)
        return 1
    print('tree %s -- identical to the local tag' % new_tree['sha'])

    print('--- committing ---')
    msg = git('tag', '-l', '--format=%(contents)', TAG).strip() or 'ICDE 2027 EAB artifact'
    commit = cli.call('POST', '/repos/%s/git/commits' % a.slug, {
        'message': ('Artifact %s: the tree the paper cites\n\n%s' % (TAG, msg)),
        'tree': tree,
        'parents': [head] if head else []})
    print('commit %s' % commit['sha'])

    print('--- moving %r ---' % branch)
    if head:
        cli.call('PATCH', '/repos/%s/git/refs/heads/%s' % (a.slug, branch),
                 {'sha': commit['sha'], 'force': True})
    else:
        cli.call('POST', '/repos/%s/git/refs' % a.slug,
                 {'ref': 'refs/heads/' + branch, 'sha': commit['sha']})
    print('%s -> %s' % (branch, commit['sha']))

    print('--- tagging %s ---' % TAG)
    tag_obj = cli.call('POST', '/repos/%s/git/tags' % a.slug, {
        'tag': TAG, 'message': msg or 'ICDE 2027 EAB artifact',
        'object': commit['sha'], 'type': 'commit'})
    try:
        cli.call('POST', '/repos/%s/git/refs' % a.slug,
                 {'ref': 'refs/tags/' + TAG, 'sha': tag_obj['sha']})
    except RuntimeError:
        cli.call('PATCH', '/repos/%s/git/refs/tags/%s' % (a.slug, TAG),
                 {'sha': tag_obj['sha'], 'force': True})
    print('tag %s -> %s' % (TAG, tag_obj['sha']))

    print('\n--- verify ---')
    remote_tree = cli.call('GET', '/repos/%s/git/trees/%s' % (a.slug, TAG))['sha']
    print('local  %s\nremote %s' % (tree, remote_tree))
    if remote_tree != tree:
        print('FAIL: the tag does not resolve to the reviewed tree')
        return 1
    print('OK   https://github.com/%s/tree/%s' % (a.slug, TAG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
