# persona_engine.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional, Literal

import numpy as np
#from sentence_transformers import SentenceTransformer

from inference import (
    compute_centroids_for_inference,
    compute_factor_pos_calibration,
    compute_factor_scores_for_texts,
    sample_texts_for_profile_simple,
)
import utils as U

# -------------------------
# Defaults / constants
# -------------------------

FACTOR_NAME_MAP: Dict[str, str] = {
    "REF": "Reflexió",
    "FOG": "Fogalmi gondolkodás",
    "SZAM": "Gondolkodás számokban",
    "GYAK": "Gyakorlati",
    "VÁL": "Vállalkozói",
    "REN": "Rendszerező",
    "KUT": "Kutatói",
    "SZOC": "Szociális",
    "MŰV": "Művészi",
    "EMO": "Emocionalitás",
    "TEM": "Tempó",
    "RUG": "Rugalmasság",
    "CSAP": "Csapatmunka",
    "KAP": "Kapcsolódás",
    "LÁT": "Látásmód",
    "HAT": "Határozottság",
    # stílusfaktorok display nevei
    "VISSZ": "Visszahúzódó kommunikáció",
    "URA": "Uralkodó kommunikáció",
    "KER": "Kerülőutas kommunikáció",
    "KIE": "Kiegyensúlyozott kommunikáció",
    "ELK": "Konfliktuskerülés",
    "ALK": "Alkalmazkodó konfliktuskezelés",
    "VER": "Versengő konfliktuskezelés",
    "KOM": "Kompromisszumkereső stílus",
    "MEG": "Megoldásfókuszú konfliktuskezelés",
}

STYLE_SENTENCES: Dict[str, Dict[int, str]] = {
    # Kommunikációs stílus
    "VISSZ": {
        5: "Kommunikációjában jól érzékelhető a visszafogottság: inkább figyel és kérdez, ritkábban hoz erős, egyértelmű állításokat.",
        6: "Kommunikációja kifejezetten visszahúzódó; jellemzően megfigyelő pozícióból van jelen, óvatosan szólal meg és kerüli a hangsúlyos szerepet.",
        7: "Kommunikációjában dominánsan megjelenik a csendes, tartózkodó jelenlét: ritkán vállal kezdeményező, véleményformáló szerepet.",
        8: "Kommunikációja erősen visszahúzódó; szinte végig megfigyelőként marad jelen, nagyon ritkán foglal nyíltan állást vagy vállal látható szerepet.",
    },
    "URA": {
        5: "Kommunikációjában gyakran határozott és erős jelenlétű, időnként átveszi a beszélgetések irányítását.",
        6: "Kommunikációja jól érzékelhetően domináns: markáns megfogalmazásokat használ, és könnyen irányító pozícióba kerül a beszélgetésekben.",
        7: "Kommunikációjában kifejezetten uralkodó stílus jelenik meg; határozottan viszi a beszélgetéseket és erősen formálja a döntési helyzeteket.",
        8: "Kommunikációja nagyon domináns, erősen alakítja a csoportdinamikát, és rendszerint ő az, aki meghatározza a beszélgetések irányát.",
    },
    "KER": {
        5: "Kommunikációjában gyakran megjelennek finomabb, indirekt megfogalmazások, időnként célozgatva jelzi a véleményét.",
        6: "Kommunikációjára jellemző a burkolt, kerülőutas stílus: üzeneteit sokszor diplomatikusan becsomagolva fogalmazza meg.",
        7: "Kommunikációjában erősen jelen van a kerülőutas működés: indirekt jelzések, célozgatás és taktikusan adagolt információk kísérik a mondandóját.",
        8: "Kommunikációja kifejezetten kerülőutas; nyílt kimondás helyett gyakran célozgat, hallgatással vagy finom jelzésekkel fejezi ki az álláspontját.",
    },
    "KIE": {
        5: "Kommunikációjában érzékelhető az asszertív, kiegyensúlyozott hang: igyekszik tényszerű, nyugodt módon érvelni.",
        6: "Kommunikációja alapvetően asszertív és nyugodt; tiszteletteljes, konstruktív hangnemben fejezi ki a véleményét és bevonja a másik felet.",
        7: "Kommunikációjában erősen jelen van az asszertív, kiegyensúlyozott működés: tárgyszerű, ítélkezésmentes, és tudatosan teret ad a párbeszédnek.",
        8: "Kommunikációja kimondottan kiegyensúlyozott és érett; magabiztosan, nyugodtan érvel, miközben következetesen figyel a másik fél szempontjaira is.",
    },

    # Konfliktuskezelési stílus
    "ELK": {
        5: "Konfliktushelyzetekben gyakran halogatja a nyílt szembenézést, és inkább kitér a feszültséget okozó témák elől.",
        6: "Konfliktuskezelésében erősen jelen van az elkerülés: sokszor kivonul a nehezebb helyzetekből, vagy igyekszik nem tudomást venni a problémáról.",
        7: "Konfliktusokban kifejezetten kerüli a konfrontációt; ritkán vállal nyílt állásfoglalást, inkább elhúzza vagy elengedi a helyzeteket.",
        8: "Konfliktuskezelésében nagyon markáns az elkerülés: a nehéz helyzeteket rendszerint elodázza vagy kikerüli, így a feszültségek könnyen bent maradnak a rendszerben.",
    },
    "ALK": {
        5: "Konfliktushelyzetekben hajlamos engedni a saját szempontjaiból a kapcsolat megőrzése érdekében.",
        6: "Konfliktuskezelésében erősen jelen van az alkalmazkodás: sokszor a másik megoldását fogadja el, hogy a viszony harmonikus maradjon.",
        7: "Konfliktusokban kifejezetten kapcsolatvédő; gyakran háttérbe helyezi saját érdekeit, csak hogy elkerülje a tartós feszültséget.",
        8: "Konfliktuskezelésében nagyon erős az önfeladó alkalmazkodás: rendszerint lemond a saját szempontjairól, ha ezzel békét tud fenntartani.",
    },
    "VER": {
        5: "Konfliktushelyzetekben határozottan képviseli a saját álláspontját, és nem riad vissza attól, hogy ezt ütköztesse másokéval.",
        6: "Konfliktuskezelésében erősen jelen van a versengő stílus: aktívan törekszik a saját érdekei érvényesítésére.",
        7: "Konfliktusokban kifejezetten versengő módon működik; erősen nyomja a saját megoldását, és nehezen enged a pozíciójából.",
        8: "Konfliktushelyzetekben nagyon domináns, versengő módon jelenik meg; markánsan a saját érdekei mentén mozgatja a helyzeteket.",
    },
    "KOM": {
        5: "Konfliktushelyzetekben törekszik arra, hogy mindkét fél számára elfogadható köztes megoldást találjon.",
        6: "Konfliktuskezelésében erősen jelen van a kompromisszumkeresés: kész engedni bizonyos pontokon, ha a másik fél is tesz lépéseket.",
        7: "Konfliktusokban tudatosan a középutat keresi; alkuképes, és figyel arra, hogy minden fél kapjon valamit a megoldásból.",
        8: "Konfliktuskezelésében nagyon erős a kompromisszumorientált működés: strukturáltan keresi a win-win megoldásokat és az egyensúlyt a felek érdekei között.",
    },
    "MEG": {
        5: "Konfliktushelyzetekben jellemző rá, hogy igyekszik a tényekre és a lehetséges megoldásokra terelni a figyelmet.",
        6: "Konfliktuskezelésében erősen jelen van a megoldásfókusz: elemző, jövőorientált módon keresi a továbblépési lehetőségeket.",
        7: "Konfliktusokban kifejezetten megoldásorientált; nem ragad le a hibakeresésnél, inkább alternatívákat épít a felek számára.",
        8: "Konfliktuskezelésében nagyon erős, érett megoldásfókusz működik: tényalapú, nyitott, win-win szemlélettel dolgozik még feszült helyzetekben is.",
    },
}

BASE_PERSONA_PROMPT_HU = """[Szerep]:
Senior teljesítménymenedzsment tanácsadóként dolgozol, sok éves B2B vezetői tanácsadói és coaching tapasztalattal, alapos pszichológiai és pszichometriai tudással.

[Segédlet az elemzéshez]:
Az alábbi tudásközpont-anyag az egyes személyiségfaktorokhoz tartozó konkrét szituációkat, szófordulatokat és fejlődési irányokat tartalmazza, az adott személy faktorértékének megfelelő polaritásban (alacsony / magas).

{QUANTILLE_BLOCK}

[Személyiségprofil – viselkedési jellemzők]:
Az alábbi lista segít jobban szűkíteni, hogy adott faktorértékhez milyen jellemzők társulhatnak, viszont nem biztos, hogy a személy pont így viselkedik, támponként használd.

{FAKTOR_BLOCK}

Munkahelyi környezet: {WORK_ENV}

[Skála]:
A faktorok 1–8 közötti skálán mozognak: 1–3 alacsony, 4–5 közepes, 6–8 magas. Közepes értéket ne írj le szélsőségként.

[Feladat]:
Készítsd el a fenti személyiségprofil összefoglaló elemzését az alábbi három szekcióban.

Stílus: elemző és lényegretörő. Röviden megfogalmazott, konkrét hétköznapi helyzetekkel szemléltesd az elemzést. Gördülékeny, igényes magyar újságírói stílus — mentes az általánosságok halmozásától. Nem patologizálsz, nem minősítesz — viselkedési mintákat írsz le.

A segédletben megadott szituációkat és szófordulatokat kiindulópontként használd, de gondold tovább konzisztensen: hozz újabb, analóg helyzeteket is, ne ragaszkodj csak a felsorolásokhoz, próbálj minél több gyakorlati példát hozni.

Egyes szám harmadik személyben írj a jelöltről!

[Elvárt kimenet]:

**1. Erősségek**
Rövid összefoglaló a személyiség mechanizmusaiban rejlő erősségekről, 2–3 konkrét hétköznapi helyzettel alátámasztva.

**2. Fejlődési lehetőségek**
*Együttműködésben:* 1–2 konkrét szituáció, ahol fejlődhet, és hogyan.
*Feladatvezérlésben:* 1–2 konkrét szituáció, ahol fejlődhet, és hogyan.

**3. Motiváció**
Mivel tudja motiválni a közvetlen vezetője? 2–3 konkrét mondat vagy helyzet.
"""

FACTORS_FILE = "../data/synthetic_generation/factors_regenerated.jsonl"
QUANTILLE_KB_FILE = "./quantille_data.json"
EngineMode = Literal["online", "artifact"]


@dataclass(frozen=True)
class SamplingConfig:
    rel_threshold: float = 0.3
    margin_threshold: float = 0.0
    top_k_factors: int = 3
    n_extreme: int = 3
    n_mid: int = 1
    rerank_by_profile: bool = True
    rerank_rel_threshold: float = 0.3
    min_rel_for_rerank: Optional[float] = 0.6
    top_n_per_bin_before_rerank: Optional[int] = 5
    bin_method: U.BinMethod = "round"


# -------------------------
# Internal helper (style snippets)
# -------------------------

def _add_style_sentences(
    out: Dict[str, Any],
    style_levels: Dict[str, int],
) -> Dict[str, Any]:
    """
    style_levels: {"VISSZ": 6, "KOM": 5, ...}
    - csak egész 1..8 jön
    - csak 5..8 esetén adunk mondatot (ha ez a szabály nálatok)
    - ha nincs mondat a dict-ben -> kimarad
    """
    for code, lvl in style_levels.items():
        if lvl < 5:
            continue

        text = STYLE_SENTENCES.get(code, {}).get(lvl)
        if not text:
            continue

        out[code] = {
            "level": float(lvl),
            "samples": [{
                "label": "STYLE",
                "bin": lvl,
                "pos": float(lvl),
                "rel": 1.0,
                "text": text,
            }],
        }

    return out

# -------------------------
# PersonaEngine
# -------------------------

class PersonaEngine:
    """
    Engine = cache + sampling + prompt builder.

    mode="online": model -> centroids -> pos_calib -> anchors -> anchor_scores
    mode="artifact": load (centroids, pos_calib, anchors, anchor_scores)

    NOTE: runtime/endpoint use-case: artifact mode recommended.
    """

    def __init__(
        self,
        mode: EngineMode = "online",
        model_path: Optional[str] = None,
        factors_file: str = FACTORS_FILE,
        artifact_path: Optional[str] = None,
        quantille_kb_path: Optional[str] = QUANTILLE_KB_FILE,
        centroid_mode: U.CentroidMode = "passage_mean",
        batch_size: int = 64,
        progress: bool = True,
    ):
        self.mode = mode
        self.factors_file = factors_file
        self.centroid_mode = centroid_mode
        self.batch_size = int(batch_size)
        self.progress = bool(progress)

        self.model: Optional[str] = None
        self.centroids: Dict[str, np.ndarray] = {}
        self.pos_calib: Dict[str, Tuple[float, float]] = {}
        self.anchor_texts: List[str] = []
        self.anchor_scores: List[Dict[str, Dict[str, float]]] = []
        self.quantille_kb: Dict[str, Any] = {}

        if quantille_kb_path:
            kb_path = Path(quantille_kb_path)
            if kb_path.exists():
                with open(kb_path, encoding="utf-8") as f:
                    self.quantille_kb = json.load(f)

        if mode == "online":
            if model_path is None:
                raise ValueError("mode='online' esetén kötelező a model_path paraméter.")
            self._init_online(model_path=model_path)
        elif mode == "artifact":
            if artifact_path is None:
                raise ValueError("mode='artifact' esetén kötelező az artifact_path paraméter.")
            self._load_artifact(artifact_path)
        else:
            raise ValueError(f"Ismeretlen mode: {mode}")

    # ---------- online init ----------

    def _init_online(self, model_path: str) -> None:
        self.model = SentenceTransformer(model_path)

        self.centroids = compute_centroids_for_inference(
            model=self.model,
            jsonl_path=self.factors_file,
            centroid_mode=self.centroid_mode,
            batch_size=self.batch_size,
            progress=self.progress,
        )

        self.pos_calib = compute_factor_pos_calibration(
            model=self.model,
            centroids=self.centroids,
            jsonl_path=self.factors_file,
            low_target=2.0,
            high_target=7.0,
        )

        self.anchor_texts = self._load_anchor_texts(self.factors_file)
        self.anchor_scores = compute_factor_scores_for_texts(
            model=self.model,
            centroids=self.centroids,
            texts=self.anchor_texts,
            pos_calib=self.pos_calib,
        )

    def _load_anchor_texts(self, path: str) -> List[str]:
        raw = U.load_raw(path)
        index = U.build_factor_index(raw)  # {factor: {LOW: [...], HIGH: [...]}}

        texts: List[str] = []
        for f in sorted(index.keys()):
            texts.extend(index[f][U.LOW])
            texts.extend(index[f][U.HIGH])

        return texts

    # ---------- artifact ----------

    def save_artifact(self, path: str) -> None:
        if not self.centroids:
            raise ValueError("Nincsenek centroids – online init futott?")
        if not self.anchor_texts or not self.anchor_scores:
            raise ValueError("Hiányos anchor_texts / anchor_scores – online módban futtatva?")

        np.savez_compressed(
            path,
            centroids=self.centroids,
            pos_calib=self.pos_calib,
            anchor_texts=np.array(self.anchor_texts, dtype=object),
            anchor_scores=np.array(self.anchor_scores, dtype=object),
        )

    def _load_artifact(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.centroids = data["centroids"].item()
        self.pos_calib = data["pos_calib"].item()
        self.anchor_texts = list(data["anchor_texts"].tolist())
        self.anchor_scores = list(data["anchor_scores"].tolist())
        self.model = None

    # ---------- core outputs for API ----------

    def build_snippets(
        self,
        profile_levels: Dict[str, int],
        cfg: SamplingConfig = SamplingConfig(),
        style_levels: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Determinisztikus “retrieval” output: factor -> {level, samples[]}
        """
        samples = sample_texts_for_profile_simple(
            profile_levels=profile_levels,
            texts=self.anchor_texts,
            all_scores=self.anchor_scores,
            rel_threshold=cfg.rel_threshold,
            margin_threshold=cfg.margin_threshold,
            top_k_factors=cfg.top_k_factors,
            n_extreme=cfg.n_extreme,
            n_mid=cfg.n_mid,
            rerank_by_profile=cfg.rerank_by_profile,
            rerank_rel_threshold=cfg.rerank_rel_threshold,
            min_rel_for_rerank=cfg.min_rel_for_rerank,
            top_n_per_bin_before_rerank=cfg.top_n_per_bin_before_rerank,
            bin_method=cfg.bin_method,
        )

        out: Dict[str, Any] = {}
        for factor, lvl in profile_levels.items():
            factor_samples = samples.get(factor, [])
            out[factor] = {
                "level": float(lvl),
                "samples": [
                    {
                        "label": label,
                        "bin": int(bin_id),
                        "pos": float(fs["pos"]),
                        "rel": float(fs["rel"]),
                        "text": text,
                    }
                    for (text, bin_id, fs, label) in factor_samples
                ],
            }

        if style_levels:
            out = _add_style_sentences(out, style_levels)

        return out
    
    def build_quantille_block(
        self,
        profile_levels: Dict[str, int],
        max_items_per_field: int = 4,
    ) -> str:
        """
        Faktorkénként kikeresi a KB-ból a releváns polaritás (low ≤ 4, high ≥ 5)
        szituációit, szófordulatait és fejlődési lehetőségeit, és egy strukturált
        szöveggé fűzi össze az LLM számára.
        """
        if not self.quantille_kb:
            return "(Nincs betöltött tudásközpont-adat.)"

        FIELD_LABELS = {
            "comfort_situations":        "Komfortos helyzetek",
            "discomfort_situations":     "Diszkomfortos helyzetek",
            "typical_phrases":           "Jellemző szóhasználat",
            "manager_motivation_phrases":"A vezető motiválhatja",
            "development_opportunities": "Fejlődési lehetőségek",
        }

        blocks: List[str] = []
        for factor_code, level in profile_levels.items():
            kb_entry = self.quantille_kb.get(factor_code)
            if kb_entry is None:
                continue

            polarity = "low" if level <= 4 else "high"
            polarity_label = "alacsony" if polarity == "low" else "magas"
            name = kb_entry.get("factor_name", factor_code)
            definition = (kb_entry.get("factor_description") or "").strip()
            pol_data = kb_entry.get(polarity, {})

            lines = [f"=== {name} ({factor_code}) | {polarity_label} ==="]
            if definition:
                lines.append(f"Definíció: {definition}")

            for field, label in FIELD_LABELS.items():
                items = pol_data.get(field, [])
                if not items:
                    continue
                lines.append(f"[{label}]")
                for item in items[:max_items_per_field]:
                    lines.append(f"  • {item}")

            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def render_factors_for_prompt(
        self,
        factor_snippets: Dict[str, Any],
        factor_name_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        LLM-nek szánt faktor-blokk:

        === GYAK (level=6.0) ===
        [HIGH] bin=6, pos=6.44, rel=0.90 | ...
        ...
        """
        lines: List[str] = []

        for factor, info in factor_snippets.items():
            level = info["level"]
            samples = info["samples"]

            if factor_name_map:
                display_name = factor_name_map.get(factor, factor)
            else:
                display_name = factor

            lines.append(f"=== {display_name} (level={level:.1f}) ===")
            for s in samples:
                label = s["label"]
                bin_id = s["bin"]
                pos = s["pos"]
                rel = s["rel"]
                text = s["text"]
                lines.append(f"  [{label}] bin={bin_id}, pos={pos:.2f}, rel={rel:.2f} | {text}")
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(
        self,
        factor_snippets: Dict[str, Any],
        profile_levels: Dict[str, int],
        work_env: str = "iroda",
        factor_name_map: Optional[Dict[str, str]] = None,
    ) -> str:
        FAKTOR_BLOCK = self.render_factors_for_prompt(factor_snippets, factor_name_map=factor_name_map)
        QUANTILLE_BLOCK = self.build_quantille_block(profile_levels)
        return BASE_PERSONA_PROMPT_HU.format(
            FAKTOR_BLOCK=FAKTOR_BLOCK,
            QUANTILLE_BLOCK=QUANTILLE_BLOCK,
            WORK_ENV=work_env,
        )

    def generate_prompt_and_snippets(
        self,
        profile_levels: Dict[str, int],
        work_env: str = "iroda",
        factor_name_map: Optional[Dict[str, str]] = None,
        style_levels: Optional[Dict[str, int]] = None,
        cfg: SamplingConfig = SamplingConfig(),
    ) -> Tuple[str, Dict[str, Any]]:
        snippets = self.build_snippets(profile_levels=profile_levels, cfg=cfg, style_levels=style_levels)
        prompt = self.build_prompt(
            snippets,
            profile_levels=profile_levels,
            work_env=work_env,
            factor_name_map=factor_name_map,
        )
        return prompt, snippets

