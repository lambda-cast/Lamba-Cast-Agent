"""
platform_db.py
==============
Manages the platform-level SQLite database (data/platform.db).

This database stores users, organizations, solar installations, and
cached PVGIS results.  It is completely separate from the existing
data/db.sqlite (telemetry) so the original Sunalyzer monitoring
functionality is not touched.

Schema
------
organizations   – companies / entities that own installations
users           – platform accounts (ADMIN or USER role)
installations   – solar PV installation records with full metadata
pvgis_cache     – cached PVGIS API responses per installation

PostGIS readiness note
----------------------
latitude/longitude are stored as REAL (float64).  The schema uses
standard column names so a future migration to PostGIS can add a
GEOMETRY column via:
    ALTER TABLE installations ADD COLUMN geom GEOMETRY(Point, 4326);
    UPDATE installations SET geom = ST_SetSRID(ST_MakePoint(longitude, latitude), 4326);
No PostGIS-specific types are used yet because SQLite does not support them.
"""

import os
import sqlite3
import logging
from pathlib import Path

# Allow tests (and production deployments) to override the DB path via env var
_DEFAULT_DB_PATH = "/home/user/LambdaCast/data/platform.db"


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
-- -----------------------------------------------------------------------
-- organizations
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS organizations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    address     TEXT,
    country     TEXT    DEFAULT 'Tunisia',
    created_at  TEXT    DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- users
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    NOT NULL UNIQUE,
    email           TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    role            TEXT    NOT NULL DEFAULT 'USER',  -- 'ADMIN' | 'USER'
    organization_id INTEGER REFERENCES organizations(id) ON DELETE SET NULL,
    is_active       INTEGER NOT NULL DEFAULT 1,        -- 0 = disabled
    created_at      TEXT    DEFAULT (datetime('now')),
    last_login      TEXT
);

-- -----------------------------------------------------------------------
-- installations
-- Full location fields for Tunisia administrative divisions.
-- PostGIS-ready: add GEOMETRY column in a future migration without
-- changing any of these columns.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS installations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    name                    TEXT    NOT NULL,
    owner_user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id         INTEGER REFERENCES organizations(id) ON DELETE SET NULL,

    -- Location — GIS foundation (ready for Leaflet map + future PostGIS)
    latitude                REAL,       -- WGS-84 decimal degrees, -90..90
    longitude               REAL,       -- WGS-84 decimal degrees, -180..180
    address                 TEXT,       -- street / building
    city                    TEXT,
    governorate             TEXT,       -- Tunisia: wilaya (e.g. "Tunis", "Sfax")
    delegation              TEXT,       -- Tunisia: mu'tamadiyya sub-district
    region                  TEXT,       -- legacy / fallback region field
    country                 TEXT    DEFAULT 'Tunisia',

    -- PV System capacity
    installed_capacity_kwp  REAL,

    -- Panel configuration
    panel_manufacturer      TEXT,
    panel_model             TEXT,
    panel_technology        TEXT,    -- monocrystalline | polycrystalline | thin_film | bifacial
    panel_power_wp          REAL,    -- Wp per panel
    number_of_panels        INTEGER,

    -- Mounting / orientation
    tilt                    REAL,    -- degrees from horizontal (0=flat, 90=vertical)
    azimuth                 REAL,    -- degrees clockwise from north (180=south)

    -- Inverter configuration
    inverter_manufacturer   TEXT,
    inverter_model          TEXT,
    inverter_capacity_kw    REAL,

    -- Administration
    installation_date       TEXT,    -- ISO date string YYYY-MM-DD
    status                  TEXT    DEFAULT 'active',  -- active | inactive | maintenance
    notes                   TEXT,

    -- Device integration (grabber plugin config)
    device_type             TEXT,    -- plugin name: iSolarCloud | Fronius | Sunsynk | Dummy | null
    device_params           TEXT,    -- JSON object with plugin-specific params (bridge_url, plant_id, etc.)

    created_at              TEXT    DEFAULT (datetime('now')),
    updated_at              TEXT    DEFAULT (datetime('now'))
);

-- Trigger to auto-update updated_at on installation changes
CREATE TRIGGER IF NOT EXISTS installations_updated_at
    AFTER UPDATE ON installations
    FOR EACH ROW
BEGIN
    UPDATE installations SET updated_at = datetime('now') WHERE id = OLD.id;
END;

-- -----------------------------------------------------------------------
-- pvgis_cache
-- Stores the parsed PVGIS API response for each installation.
-- One row per installation (upserted on refresh).
-- The raw_response column preserves the complete PVGIS JSON for auditing.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pvgis_cache (
    installation_id     INTEGER PRIMARY KEY
                        REFERENCES installations(id) ON DELETE CASCADE,

    -- Parsed summary values (fast access without JSON parsing)
    annual_production_kwh   REAL,           -- kWh/year
    specific_yield_kwh_kwp  REAL,           -- kWh/kWp/year
    performance_ratio       REAL,           -- 0..1
    irradiation_kwh_m2_year REAL,           -- kWh/m²/year (global tilted irradiance)

    -- Monthly breakdown stored as a JSON array [Jan..Dec] in kWh
    monthly_production_json TEXT,           -- e.g. [120.1, 135.2, ..., 110.0]

    -- PVGIS parameters used for this calculation (JSON object)
    pvgis_params_json       TEXT,

    -- Metadata
    pvgis_database          TEXT    DEFAULT 'PVGIS-SARAH2',
    calculated_at           TEXT    DEFAULT (datetime('now')),
    pvgis_api_version       TEXT    DEFAULT 'v5_2',

    -- Full raw response (for debugging / re-parsing without re-calling PVGIS)
    raw_response_json       TEXT
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class PlatformDatabase:
    """
    Thin wrapper around an SQLite connection for the platform tables.

    Each method opens and closes its own connection to stay compatible with
    the multi-process (grabber + server) deployment model used by Sunalyzer.
    """

    def __init__(self, db_path: str | None = None):
        # Priority: explicit arg > env var > default
        self.db_path = db_path or os.environ.get("SUNALYZER_PLATFORM_DB", _DEFAULT_DB_PATH)
        self._ensure_schema()
        self._ensure_device_columns()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Returns a new connection with row_factory set to dict-like rows."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _ensure_schema(self):
        """Creates tables if they don't exist yet."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
        logging.info(f"PlatformDatabase: schema ready at {self.db_path}")

    def _ensure_device_columns(self):
        """Add device_type / device_params if this is an older DB missing them."""
        with self._connect() as conn:
            info = [r[1] for r in conn.execute("PRAGMA table_info(installations)").fetchall()]
            if "device_type" not in info:
                conn.execute("ALTER TABLE installations ADD COLUMN device_type TEXT")
                logging.info("PlatformDatabase: added device_type column")
            if "device_params" not in info:
                conn.execute("ALTER TABLE installations ADD COLUMN device_params TEXT")
                logging.info("PlatformDatabase: added device_params column")
            conn.commit()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def execute(self, sql: str, params: tuple = ()) -> int:
        """Execute a DML statement; returns lastrowid."""
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur.lastrowid

    # ------------------------------------------------------------------
    # User operations
    # ------------------------------------------------------------------

    def list_installations_for_user(self, user_id: int) -> list[dict]:
        """User view — only their own installations."""
        return self.fetchall(
            "SELECT * FROM installations WHERE owner_user_id = ? ORDER BY id",
            (user_id,),
        )
