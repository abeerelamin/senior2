import json
import os

import ee


EE_READY = False
EE_ERROR = None


def init_ee():
    global EE_READY, EE_ERROR
    if EE_READY:
        return

    try:
        service_account_json = (os.environ.get("EE_SERVICE_ACCOUNT_JSON") or "").strip()
        if not service_account_json:
            raise RuntimeError(
                "EE_SERVICE_ACCOUNT_JSON is missing. "
                "Set it in Render → Environment (full JSON key as one line or escaped)."
            )

        info = json.loads(service_account_json)
        email = info["client_email"]
        # Prefer EE_PROJECT if set (Render/GCP); else project_id inside the key JSON
        project_id = (os.environ.get("EE_PROJECT") or "").strip() or info.get("project_id")

        credentials = ee.ServiceAccountCredentials(email, key_data=service_account_json)
        if project_id:
            ee.Initialize(credentials, project=project_id)
        else:
            ee.Initialize(credentials)

        EE_READY = True
        EE_ERROR = None
        print("Earth Engine initialized successfully.", "project=", project_id or "(default)")
    except Exception as e:
        EE_READY = False
        EE_ERROR = str(e)
        print("Failed to initialize Earth Engine:", EE_ERROR)
