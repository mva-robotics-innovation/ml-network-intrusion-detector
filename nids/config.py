"""Central configuration: paths, schema, label taxonomy, branding."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.getenv("NIDS_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
DATA_DIR = Path(os.getenv("NIDS_DATA_DIR", ROOT_DIR / "data"))

MODEL_FILE = "nids_model.pkl"
METRICS_FILE = "metrics.json"
CONFUSION_PNG = "confusion_matrix.png"
IMPORTANCE_CSV = "feature_importance.csv"

TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"

RANDOM_STATE = 42

# --- Branding / provenance -------------------------------------------------
PRODUCT_NAME = "ML-NIDS Enterprise"
DEVELOPER = "Ashish Kumar"
AFFILIATION = "Student-led startup of Jharkhand Rai University"
ORGANIZATION = "Director & Founder, MVA Robotics Innovation Org"
PROGRAM = "IBM Cybersecurity Bootcamp 2026"

# --- NSL-KDD schema (41 features) -----------------------------------------
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]
assert len(FEATURE_NAMES) == 41

# Defaults used when a caller supplies only a subset of the 41 features.
DEFAULT_FLOW = {f: 0.0 for f in NUMERIC_FEATURES}
DEFAULT_FLOW.update({
    "src_bytes": 500.0, "dst_bytes": 1000.0, "logged_in": 1.0,
    "count": 10.0, "srv_count": 10.0, "same_srv_rate": 1.0,
    "dst_host_count": 100.0, "dst_host_srv_count": 100.0,
    "dst_host_same_srv_rate": 1.0,
})
DEFAULT_FLOW.update({"protocol_type": "tcp", "service": "http", "flag": "SF"})

# Vocabulary used ONLY by the synthetic fallback generator.
SYNTH_PROTOCOLS = ["tcp", "udp", "icmp"]
SYNTH_SERVICES = ["http", "ftp", "smtp", "ssh", "telnet", "domain_u",
                  "private", "ecr_i", "other", "auth"]
SYNTH_FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "SH"]

# --- Label taxonomy --------------------------------------------------------
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

ATTACK_CATEGORY = {"normal": "Normal"}
for _name in ("back", "land", "neptune", "pod", "smurf", "teardrop",
              "apache2", "udpstorm", "processtable", "mailbomb", "worm"):
    ATTACK_CATEGORY[_name] = "DoS"
for _name in ("satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"):
    ATTACK_CATEGORY[_name] = "Probe"
for _name in ("guess_passwd", "ftp_write", "imap", "phf", "multihop",
              "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
              "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"):
    ATTACK_CATEGORY[_name] = "R2L"
for _name in ("buffer_overflow", "loadmodule", "rootkit", "perl",
              "sqlattack", "xterm", "ps"):
    ATTACK_CATEGORY[_name] = "U2R"

# Operational policy (edit to match your SOC runbooks).
THREAT_LEVELS = {
    "Normal": "None", "DoS": "Critical", "U2R": "High",
    "R2L": "Medium", "Probe": "Medium",
}
SEVERITY_SCORE = {"None": 0, "Medium": 50, "High": 75, "Critical": 100}


def footer_text(version: str) -> str:
    return (f"{PRODUCT_NAME} v{version} | Developer: {DEVELOPER} | "
            f"{AFFILIATION} | {ORGANIZATION} | {PROGRAM}")
