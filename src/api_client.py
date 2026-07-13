"""

    POST /auth/login
    GET  /ml/bearings  -- endpoint one where
    GET  /ml/bearings/{bearingLocationId}/raw  --- api end point 2

Performance notes (this is the client used for daily bulk fetching, which can
mean tens of thousands of requests):
- Uses a single persistent requests.Session with a large connection pool,
  instead of the module-level requests.get()/post() functions. Plain
  requests.get() opens a fresh TCP+TLS connection for every call — with a
  large bearing fleet fetched via a thread pool, that's the single biggest
  source of unnecessary latency, since every request pays a full handshake
  instead of reusing an already-open keep-alive connection.
- pool_maxsize should be set >= your ThreadPoolExecutor's max_workers (see
  src/fetch_live_data.py), or threads will block waiting for a free pooled
  connection instead of actually running in parallel.
- list_bearings() defaults to NOT passing measuringType at all — per the API
  doc, omitting it returns bearings of every type in a single paginated walk,
  instead of looping once per type (sensor/vibrolink/turbine/multichannel)
  and re-paginating each time.
"""
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class AAMSClient:
    def __init__(self, base_url=None, email=None, password=None, pool_maxsize=50):
        self.base_url = base_url or config.AAMS_BASE_URL
        self.email = email or config.AAMS_EMAIL
        self.password = password or config.AAMS_PASSWORD
        self._token = None

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize, max_retries=2)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def login(self):
        if not self.email or not self.password:
            raise ValueError(
                "AAMS_EMAIL / AAMS_PASSWORD are not set. Export them as environment "
                "variables, put them in a .env file, or set them in config.py before "
                "calling the live API."
            )
        r = self._session.post(
            f"{self.base_url}/auth/login",
            json={"email": self.email, "password": self.password},
            timeout=30,
        )
        r.raise_for_status()
        self._token = r.json()["access_token"]
        return self._token

    def _headers(self):
        if not self._token:
            self.login()
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, path, params=None, retry_on_401=True):
        r = self._session.get(f"{self.base_url}{path}", headers=self._headers(), params=params, timeout=60)
        if r.status_code == 401 and retry_on_401:
            self.login()
            return self._get(path, params=params, retry_on_401=False)
        r.raise_for_status()
        return r.json()

    def list_bearings(self, measuring_type=None, customer_id=None, alert_level=None, page_size=2000):
        """
        Page through /ml/bearings and return the full list of bearing records.
        measuring_type defaults to None (omitted) -> returns ALL measuring
        types in one paginated walk, per the API's documented default. Pass
        an explicit type only if you specifically want to filter to one.
        page_size defaults to 2000 (the documented max) to minimize the
        number of page round-trips.
        """
        bearings, page = [], 1
        while True:
            params = {"page": page, "pageSize": page_size}
            if measuring_type:
                params["measuringType"] = measuring_type
            if customer_id:
                params["customerId"] = customer_id
            if alert_level:
                params["alertLevel"] = alert_level

            resp = self._get("/ml/bearings", params=params)
            bearings.extend(resp["data"])
            if page >= resp["totalPages"]:
                break
            page += 1
        return bearings

    def get_raw(self, bearing_location_id, date=None, start_date=None, end_date=None,
                axis=None, analytics_type=None):
        """Fetch raw waveform packets for one bearing, for a single date or a range (<=31 days)."""
        params = {}
        if start_date and end_date:
            params["startDate"] = start_date
            params["endDate"] = end_date
        else:
            params["date"] = date or time.strftime("%Y-%m-%d", time.gmtime())
        if axis:
            params["axis"] = axis
        if analytics_type:
            params["analyticsType"] = analytics_type

        return self._get(f"/ml/bearings/{bearing_location_id}/raw", params=params)
