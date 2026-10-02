import urllib.request
import json
import time

def api_post(endpoint, payload=None):
    url = f"http://127.0.0.1:8001{endpoint}"
    data = json.dumps(payload or {}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def api_get(endpoint):
    url = f"http://127.0.0.1:8001{endpoint}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode())

print("1. Initiating training...")
start_res = api_post("/train/start", {"architecture": "swin_transformer", "epochs": 2, "sync_dataset": False})
print("Start response:", start_res)

time.sleep(1.0)
st = api_get("/train/status")
print("Status after 1s:", st["status"], "| progress:", st["progress"])

print("2. Requesting cancellation...")
t0 = time.time()
cancel_res = api_post("/train/cancel")
print("Cancel response:", cancel_res)

# Poll until cancelled
while True:
    st = api_get("/train/status")
    print(f"Polling status ({time.time() - t0:.2f}s):", st["status"], "| msg:", st["message"])
    if st["status"] == "cancelled":
        print(f"SUCCESS: Aborted cleanly in {time.time() - t0:.2f}s!")
        break
    if time.time() - t0 > 15:
        print("FAIL: Timed out waiting for cancellation!")
        break
    time.sleep(0.5)
