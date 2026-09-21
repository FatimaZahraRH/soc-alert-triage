"""
Générateur de flux réseau synthétiques, avec un schéma de caractéristiques
inspiré du dataset public CICIDS2017 (Sharafaldin et al., 2018) largement
utilisé dans la littérature SOC/détection d'intrusion (dont le papier
ExTriage cité en référence de ce projet).

Le dataset réel n'est pas téléchargé (réseau restreint dans cet
environnement + volumétrie), mais la structure des caractéristiques et les
signatures statistiques par type d'attaque reproduisent fidèlement les
grandes lignes documentées dans la littérature.

Point de conception clé, cohérent avec mes projets précédents (Mini-DSPM,
RAG Red-Team) : le corpus inclut des "cas difficiles" volontaires --
du trafic bénin qui ressemble structurellement à une attaque (gros transfert
de sauvegarde légitime, scan de vulnérabilité autorisé par l'équipe
sécurité) -- sans quoi l'évaluation ROC serait artificiellement parfaite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, dataset_config

FEATURE_COLUMNS = [
    "flow_duration_ms", "total_fwd_packets", "total_bwd_packets",
    "total_fwd_bytes", "total_bwd_bytes", "flow_bytes_per_s", "flow_packets_per_s",
    "fwd_packet_length_mean", "bwd_packet_length_mean", "fwd_iat_mean_ms",
    "syn_flag_count", "fin_flag_count", "dest_port", "protocol",
]

_PROTOCOLS = {"TCP": 6, "UDP": 17, "ICMP": 1}
_COMMON_PORTS = [80, 443, 22, 3389, 53, 8080, 445, 3306]

_rng = np.random.default_rng(RANDOM_SEED)


def _clip_nonneg(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, a_min=0, a_max=None)


def _gen_benign(n: int, hard_negative: bool = False) -> pd.DataFrame:
    if not hard_negative:
        duration = _rng.gamma(shape=3.0, scale=800, size=n)
        fwd_pkts = _rng.poisson(15, size=n) + 1
        bwd_pkts = _rng.poisson(14, size=n) + 1
        fwd_bytes = fwd_pkts * _rng.normal(600, 150, size=n)
        bwd_bytes = bwd_pkts * _rng.normal(800, 200, size=n)
        syn = _rng.integers(1, 3, size=n)
        fin = _rng.integers(1, 3, size=n)  # terminaison propre
        ports = _rng.choice(_COMMON_PORTS, size=n)
    else:
        # Cas difficile n°1 : gros transfert de sauvegarde légitime --
        # volumétrie élevée comme une attaque volumétrique, mais terminaison
        # TCP propre (fin_flag présent) et port de service attendu.
        duration = _rng.gamma(shape=4.0, scale=4000, size=n)
        fwd_pkts = _rng.poisson(800, size=n) + 50
        bwd_pkts = _rng.poisson(600, size=n) + 30
        fwd_bytes = fwd_pkts * _rng.normal(1200, 200, size=n)
        bwd_bytes = bwd_pkts * _rng.normal(400, 100, size=n)
        syn = _rng.integers(1, 3, size=n)
        fin = _rng.integers(1, 3, size=n)  # signature de terminaison propre = distingue de DoS
        ports = _rng.choice([443, 22], size=n)

    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "BENIGN")


def _gen_portscan(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=1.2, scale=15, size=n)  # très court
    fwd_pkts = _rng.integers(1, 4, size=n)
    bwd_pkts = _rng.integers(0, 2, size=n)  # peu/pas de réponse
    fwd_bytes = fwd_pkts * _rng.normal(60, 15, size=n)
    bwd_bytes = bwd_pkts * _rng.normal(40, 10, size=n)
    syn = _rng.integers(1, 3, size=n)
    fin = np.zeros(n)  # pas de terminaison propre
    ports = _rng.integers(1, 65535, size=n)  # ports aléatoires balayés
    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "PortScan")


def _gen_portscan_authorized(n: int) -> pd.DataFrame:
    """Cas difficile n°2 : scan de vulnérabilité autorisé par l'équipe
    sécurité interne -- signature réseau quasi identique à un vrai PortScan
    (c'est un vrai scan), mais c'est un flux BÉNIN du point de vue métier
    (contexte : asset_criticality / source de confiance, géré par le module
    de triage, pas par le classifieur réseau seul)."""
    df = _gen_portscan(n)
    df["label"] = "BENIGN"
    df["known_authorized_source"] = True
    return df


def _gen_web_bruteforce(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=2.0, scale=200, size=n)
    fwd_pkts = _rng.poisson(25, size=n) + 5  # tentatives répétées
    bwd_pkts = _rng.poisson(20, size=n) + 5
    fwd_bytes = fwd_pkts * _rng.normal(300, 60, size=n)
    bwd_bytes = bwd_pkts * _rng.normal(250, 50, size=n)
    syn = _rng.integers(3, 8, size=n)  # connexions répétées
    fin = _rng.integers(0, 2, size=n)
    ports = _rng.choice([80, 443], size=n)
    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "WebAttack_BruteForce")


def _gen_bot(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=2.5, scale=600, size=n)
    fwd_pkts = _rng.poisson(8, size=n) + 2  # petit trafic périodique (beaconing)
    bwd_pkts = _rng.poisson(6, size=n) + 1
    fwd_bytes = fwd_pkts * _rng.normal(150, 30, size=n)
    bwd_bytes = bwd_pkts * _rng.normal(100, 20, size=n)
    syn = _rng.integers(1, 3, size=n)
    fin = _rng.integers(0, 2, size=n)
    ports = _rng.choice([8080, 6667, 4444, 53], size=n)  # ports C2 typiques
    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "Bot")


def _gen_dos_hulk(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=1.5, scale=100, size=n)
    fwd_pkts = _rng.poisson(500, size=n) + 100  # volumétrie très élevée
    bwd_pkts = _rng.integers(0, 5, size=n)  # cible débordée, quasi aucune réponse
    fwd_bytes = fwd_pkts * _rng.normal(80, 20, size=n)  # petits paquets répétés
    bwd_bytes = bwd_pkts * _rng.normal(50, 10, size=n)
    syn = _rng.integers(20, 60, size=n)
    fin = np.zeros(n)
    ports = np.full(n, 80)
    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "DoS_Hulk")


def _gen_ddos(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=1.2, scale=50, size=n)
    fwd_pkts = _rng.poisson(1500, size=n) + 300  # volumétrie extrême
    bwd_pkts = _rng.integers(0, 3, size=n)
    fwd_bytes = fwd_pkts * _rng.normal(70, 15, size=n)
    bwd_bytes = bwd_pkts * _rng.normal(40, 10, size=n)
    syn = _rng.integers(50, 150, size=n)
    fin = np.zeros(n)
    ports = np.full(n, 443)
    protocol = np.full(n, _PROTOCOLS["UDP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "DDoS")


def _gen_infiltration(n: int) -> pd.DataFrame:
    duration = _rng.gamma(shape=5.0, scale=3000, size=n)  # longue durée, discret
    fwd_pkts = _rng.poisson(40, size=n) + 5
    bwd_pkts = _rng.poisson(200, size=n) + 20  # exfiltration -> gros trafic sortant (bwd du point de vue serveur)
    fwd_bytes = fwd_pkts * _rng.normal(200, 40, size=n)
    bwd_bytes = bwd_pkts * _rng.normal(1500, 300, size=n)  # volumétrie de données exfiltrées
    syn = _rng.integers(1, 3, size=n)
    fin = _rng.integers(0, 2, size=n)
    ports = _rng.integers(20000, 60000, size=n)  # ports hauts non standards
    protocol = np.full(n, _PROTOCOLS["TCP"])
    return _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, "Infiltration")


def _assemble(n, duration, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, syn, fin, ports, protocol, label) -> pd.DataFrame:
    duration = _clip_nonneg(duration) + 1.0  # éviter division par zéro
    fwd_bytes = _clip_nonneg(fwd_bytes)
    bwd_bytes = _clip_nonneg(bwd_bytes)
    total_bytes = fwd_bytes + bwd_bytes
    total_pkts = fwd_pkts + bwd_pkts

    flow_bytes_per_s = total_bytes / (duration / 1000.0)
    flow_packets_per_s = total_pkts / (duration / 1000.0)
    fwd_pkt_len_mean = np.where(fwd_pkts > 0, fwd_bytes / np.maximum(fwd_pkts, 1), 0)
    bwd_pkt_len_mean = np.where(bwd_pkts > 0, bwd_bytes / np.maximum(bwd_pkts, 1), 0)
    fwd_iat_mean = duration / np.maximum(fwd_pkts, 1)

    return pd.DataFrame({
        "flow_duration_ms": duration,
        "total_fwd_packets": fwd_pkts,
        "total_bwd_packets": bwd_pkts,
        "total_fwd_bytes": fwd_bytes,
        "total_bwd_bytes": bwd_bytes,
        "flow_bytes_per_s": flow_bytes_per_s,
        "flow_packets_per_s": flow_packets_per_s,
        "fwd_packet_length_mean": fwd_pkt_len_mean,
        "bwd_packet_length_mean": bwd_pkt_len_mean,
        "fwd_iat_mean_ms": fwd_iat_mean,
        "syn_flag_count": syn,
        "fin_flag_count": fin,
        "dest_port": ports,
        "protocol": protocol,
        "label": label,
    })


def generate_dataset() -> pd.DataFrame:
    cfg = dataset_config
    n_benign = cfg.n_samples_per_class["BENIGN"]
    n_hard = int(n_benign * cfg.hard_negative_ratio)
    n_benign_normal = n_benign - n_hard

    parts = [
        _gen_benign(n_benign_normal, hard_negative=False),
        _gen_benign(n_hard // 2, hard_negative=True),
        _gen_portscan_authorized(n_hard - n_hard // 2),
        _gen_portscan(cfg.n_samples_per_class["PortScan"]),
        _gen_web_bruteforce(cfg.n_samples_per_class["WebAttack_BruteForce"]),
        _gen_bot(cfg.n_samples_per_class["Bot"]),
        _gen_dos_hulk(cfg.n_samples_per_class["DoS_Hulk"]),
        _gen_ddos(cfg.n_samples_per_class["DDoS"]),
        _gen_infiltration(cfg.n_samples_per_class["Infiltration"]),
    ]
    df = pd.concat(parts, ignore_index=True)

    # Contexte métier (utilisé par le module de triage, pas par le
    # classifieur réseau) : quel type d'actif est concerné par le flux.
    df["asset_criticality"] = _rng.choice(
        ["low", "medium", "high"], size=len(df), p=[0.5, 0.35, 0.15],
    )

    df["flow_id"] = [f"FLOW-{i:06d}" for i in range(len(df))]
    df["known_authorized_source"] = df.get("known_authorized_source", False)
    df["known_authorized_source"] = df["known_authorized_source"].fillna(False).astype(bool)

    # Identifiant de source persistant : le petit pool de scanners internes
    # autorisés réutilise les mêmes identifiants (comme de vraies machines de
    # scan) ; toutes les autres sources sont uniques par flux, ce qui permet
    # de simuler un apprentissage par retour d'expérience analyste réaliste
    # (cf. src/triage/analyst_feedback.py) sans généraliser à tort sur de
    # vraies attaques, qui elles proviennent de sources diverses/inconnues.
    _authorized_pool = [f"SCANNER-{i:02d}" for i in range(1, 4)]
    source_ids = []
    for idx, is_auth in enumerate(df["known_authorized_source"]):
        if is_auth:
            source_ids.append(_authorized_pool[idx % len(_authorized_pool)])
        else:
            source_ids.append(f"SRC-{idx:06d}")
    df["source_asset_id"] = source_ids

    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def is_malicious(label: str) -> bool:
    return label != "BENIGN"
