"""Test a fresh container with no network and no host mounts; always remove the test container."""

import argparse
import json
import subprocess
import uuid

from ielts_api.artifacts import MODEL, RELEASE, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="An exact local image tag or image ID")
    args = parser.parse_args()
    fixture = json.loads((ROOT / "tests/fixtures/reference.json").read_text())
    name = f"ielts-offline-smoke-{uuid.uuid4().hex[:10]}"
    # HTTP stays inside the isolated container. Only the test program travels over stdin.
    # There are no host cache mounts and no outbound network interfaces.
    program = """
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def request(route, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request('http://127.0.0.1:3000' + route, data=data,
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        body = response.read()
        is_json = response.headers.get_content_type() == 'application/json'
        parsed = json.loads(body) if is_json else None
        return response.status, parsed

deadline = time.monotonic() + 150
while True:
    try:
        if request('/readyz')[0] == 200:
            break
    except (HTTPError, URLError, TimeoutError):
        pass
    if time.monotonic() > deadline:
        raise RuntimeError('Container did not become ready')
    time.sleep(1)
assert request('/livez')[0] == 200
assert '/score' in request('/docs.json')[1]['paths']
worst = 0.0
for case in fixture['cases']:
    status, actual = request('/score', case['input'])
    expected = case['expected']
    assert status == 200
    assert actual['release']['model_revision'] == model_revision
    assert actual['release']['application_version'] == application_version
    for key in ['track', 'max_tokens', 'truncated']:
        assert actual[key] == expected[key], key
    errors = [abs(actual['scores'][k] - expected['scores'][k]) for k in expected['scores']]
    errors.append(abs(actual['task2_mean_estimate'] - expected['task2_mean_estimate']))
    error = max(errors)
    assert error <= fixture['absolute_tolerance'], errors
    worst = max(worst, error)
    print(f"PASS offline container {case['name']}: max error={error:.8f}", flush=True)
try:
    request('/score', {'question': ' ', 'essay': 'A valid essay.'})
except HTTPError as exc:
    assert exc.code == 400, exc.code
else:
    raise AssertionError('Invalid request was accepted')
print(json.dumps({'network': 'none', 'host_mounts': 0, 'cases': len(fixture['cases']),
                  'max_absolute_error': worst, 'tolerance': fixture['absolute_tolerance']}))
"""
    injected = (
        f"fixture = {fixture!r}\n"
        f"model_revision = {MODEL['revision']!r}\n"
        f"application_version = {RELEASE['application_version']!r}\n" + program
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "--cpus",
            "2",
            "--memory",
            "3g",
            args.image,
        ],
        check=True,
    )
    try:
        state = json.loads(subprocess.check_output(["docker", "inspect", name], text=True))[0]
        assert state["State"]["Running"]
        assert state["HostConfig"]["NetworkMode"] == "none"
        assert state["Mounts"] == []
        subprocess.run(
            ["docker", "exec", "-i", name, "python", "-"], input=injected, text=True, check=True
        )
    except BaseException:
        subprocess.run(["docker", "logs", "--tail", "80", name], check=False)
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True)


if __name__ == "__main__":
    main()
