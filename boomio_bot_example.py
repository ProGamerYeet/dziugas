"""
Minimal Python example for online game bot training without desktop screenshots.

Approach:
- Control browser with Playwright.
- Extract state from page JS/DOM and network (WebSocket) frames.
- Run multiple workers in parallel with multiprocessing.
- Send transitions/events to a central trainer process through a Queue.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


GAME_URL = "https://boomio.com/zemaitijos-pienas-dziugas-lt"
DEFAULT_NUM_WORKERS = 1
DEFAULT_STEPS_PER_WORKER = 0
DEFAULT_WEIGHTS_FILE = "pixel_policy_weights.npz"
DEFAULT_LEARNING_RATE = 0.001
MOVE_TO_KEY = {
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "up": "ArrowUp",
    "down": "ArrowDown",
    "idle": None,
}
ACTION_ORDER = ["left", "right", "up", "down", "idle"]
GRID_X_MIN = 20
GRID_X_MAX = 400
GRID_Y_MIN = 355
GRID_Y_MAX = 564
GRID_COLS = 12
GRID_ROWS = 8
GRID_INPUT_DIM = GRID_COLS * GRID_ROWS * 3


class PixelPolicyNet:
    def __init__(self, weights_file: str, verbose: bool = True):
        self.weights_file = weights_file
        self.hidden = 10
        self.outputs = len(ACTION_ORDER)
        self.baseline = 0.0
        self.steps = 0
        self.verbose = verbose
        if os.path.exists(weights_file):
            data = np.load(weights_file)
            self.w1 = data["w1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.w2 = data["w2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            self.baseline = float(data["baseline"]) if "baseline" in data else 0.0
            self.steps = int(data["steps"]) if "steps" in data else 0
            self._last_mtime = os.path.getmtime(weights_file)
            if self.verbose:
                print(f"[policy] loaded weights from {weights_file}")
        else:
            rng = np.random.default_rng(seed=worker_seed_from_time())
            self.w1 = (rng.standard_normal((GRID_INPUT_DIM, self.hidden)) * 0.02).astype(np.float32)
            self.b1 = np.zeros((self.hidden,), dtype=np.float32)
            self.w2 = (rng.standard_normal((self.hidden, self.outputs)) * 0.02).astype(np.float32)
            self.b2 = np.zeros((self.outputs,), dtype=np.float32)
            self._last_mtime = 0.0
            if self.verbose:
                print(f"[policy] initialized new weights ({weights_file})")

    def save(self) -> None:
        np.savez(
            self.weights_file,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            baseline=np.array(self.baseline, dtype=np.float32),
            steps=np.array(self.steps, dtype=np.int64),
        )
        self._last_mtime = os.path.getmtime(self.weights_file)

    def reload_if_updated(self) -> bool:
        if not os.path.exists(self.weights_file):
            return False
        mtime = os.path.getmtime(self.weights_file)
        if mtime <= self._last_mtime:
            return False
        data = np.load(self.weights_file)
        self.w1 = data["w1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.w2 = data["w2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.baseline = float(data["baseline"]) if "baseline" in data else self.baseline
        self.steps = int(data["steps"]) if "steps" in data else self.steps
        self._last_mtime = mtime
        return True

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits = logits - np.max(logits)
        exp = np.exp(logits).astype(np.float32)
        probs = exp / np.sum(exp)
        return h.astype(np.float32), probs.astype(np.float32)

    def sample_action(self, x: np.ndarray) -> tuple[int, str, np.ndarray, np.ndarray]:
        h, probs = self._forward(x)
        action_idx = int(np.random.choice(self.outputs, p=probs))
        return action_idx, ACTION_ORDER[action_idx], probs, h

    def train_step(
        self, x: np.ndarray, h: np.ndarray, probs: np.ndarray, action_idx: int, reward: float, lr: float
    ) -> dict[str, float]:
        self.baseline = 0.99 * self.baseline + 0.01 * reward
        advantage = reward - self.baseline

        grad_logits = probs.copy()
        grad_logits[action_idx] -= 1.0
        grad_logits *= advantage

        grad_w2 = np.outer(h, grad_logits)
        grad_b2 = grad_logits
        grad_h = self.w2 @ grad_logits
        grad_h_pre = grad_h * (1.0 - h * h)
        grad_w1 = np.outer(x, grad_h_pre)
        grad_b1 = grad_h_pre

        self.w2 -= lr * grad_w2
        self.b2 -= lr * grad_b2
        self.w1 -= lr * grad_w1
        self.b1 -= lr * grad_b1
        self.steps += 1
        grad_norm = float(
            np.sqrt(
                np.sum(grad_w1 * grad_w1)
                + np.sum(grad_b1 * grad_b1)
                + np.sum(grad_w2 * grad_w2)
                + np.sum(grad_b2 * grad_b2)
            )
        )
        weight_norm = float(
            np.sqrt(
                np.sum(self.w1 * self.w1)
                + np.sum(self.b1 * self.b1)
                + np.sum(self.w2 * self.w2)
                + np.sum(self.b2 * self.b2)
            )
        )
        return {
            "advantage": float(advantage),
            "grad_norm": grad_norm,
            "weight_norm": weight_norm,
            "baseline": float(self.baseline),
        }


def worker_seed_from_time() -> int:
    return int(time.time() * 1_000_000) % (2**31 - 1)


def extract_grid_rgb_features(frame_png: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(frame_png)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    height, width, _ = arr.shape

    features: list[float] = []
    for r in range(GRID_ROWS):
        y = GRID_Y_MIN + (r * (GRID_Y_MAX - GRID_Y_MIN)) / (GRID_ROWS - 1)
        yi = int(np.clip(round(y), 0, height - 1))
        for c in range(GRID_COLS):
            x = GRID_X_MIN + (c * (GRID_X_MAX - GRID_X_MIN)) / (GRID_COLS - 1)
            xi = int(np.clip(round(x), 0, width - 1))
            rgb = arr[yi, xi]
            features.extend([float(rgb[0]), float(rgb[1]), float(rgb[2])])

    return np.asarray(features, dtype=np.float32)


def save_grid_debug_snapshot(page, worker_id: int, step: int) -> None:
    os.makedirs("game_snapshots", exist_ok=True)
    path = os.path.join("game_snapshots", f"worker{worker_id:02d}_step{step:06d}_grid.png")

    page.evaluate(
        """() => {
            const existing = document.getElementById("copilot-grid-overlay");
            if (existing) existing.remove();

            const root = document.getElementById("game-container");
            if (!root) return;
            if (window.getComputedStyle(root).position === "static") {
                root.style.position = "relative";
            }

            const overlay = document.createElement("div");
            overlay.id = "copilot-grid-overlay";
            overlay.style.position = "absolute";
            overlay.style.left = "0";
            overlay.style.top = "0";
            overlay.style.width = "100%";
            overlay.style.height = "100%";
            overlay.style.pointerEvents = "none";
            overlay.style.zIndex = "99999";

            const xMin = 20;
            const xMax = 400;
            const yMin = 355;
            const yMax = 564;
            const cols = 12;
            const rows = 8;

            const border = document.createElement("div");
            border.style.position = "absolute";
            border.style.left = `${xMin}px`;
            border.style.top = `${yMin}px`;
            border.style.width = `${xMax - xMin}px`;
            border.style.height = `${yMax - yMin}px`;
            border.style.border = "2px solid #00ff66";
            border.style.boxSizing = "border-box";
            overlay.appendChild(border);

            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const x = xMin + (c * (xMax - xMin)) / (cols - 1);
                    const y = yMin + (r * (yMax - yMin)) / (rows - 1);
                    const dot = document.createElement("div");
                    dot.style.position = "absolute";
                    dot.style.left = `${x - 2}px`;
                    dot.style.top = `${y - 2}px`;
                    dot.style.width = "4px";
                    dot.style.height = "4px";
                    dot.style.borderRadius = "50%";
                    dot.style.background = "#ff2d55";
                    dot.style.boxShadow = "0 0 3px #000";
                    overlay.appendChild(dot);
                }
            }

            root.appendChild(overlay);
        }"""
    )

    game_root = page.locator("#game-container").first
    if game_root.count() > 0:
        game_root.screenshot(path=path)
    else:
        page.screenshot(path=path, full_page=False)

    page.evaluate(
        """() => {
            const existing = document.getElementById("copilot-grid-overlay");
            if (existing) existing.remove();
        }"""
    )
    print(f"[debug] saved grid snapshot: {path}")


def capture_game_frame(page) -> bytes:
    game_root = page.locator("#game-container").first
    if game_root.count() > 0:
        return game_root.screenshot()
    return page.screenshot(full_page=False)


def read_state(page) -> dict[str, Any]:
    """
    Try to get useful state from DOM/JS without screenshots.
    Extend this with real game signals when discovered.
    """
    return page.evaluate(
        """() => ({
            title: document.title,
            readyState: document.readyState,
            screenText: (() => {
                const root = document.getElementById("game-container") ?? document.body;
                return (root?.innerText || "").replace(/\\s+/g, " ").trim();
            })(),
            score: (() => {
                const raw = document.getElementById("currentScore")?.textContent?.trim() ?? "";
                const digits = raw.replace(/\\D/g, "");
                if (!digits) return null;
                const parsed = Number(digits);
                return Number.isNaN(parsed) ? null : parsed;
            })(),
            livesRemaining: (() => {
                const lifeRaw = document.getElementById("currentLife")?.textContent?.trim() ?? "";
                const lifeMatch = lifeRaw.match(/([0-3])\\s*\\/\\s*3/);
                if (lifeMatch) return Number(lifeMatch[1]);

                const fromObj = [
                    window?.game?.lives,
                    window?.game?.life,
                    window?.game?.player?.lives,
                    window?.gameState?.lives,
                    window?.state?.lives,
                    window?.store?.state?.lives,
                ];
                for (const v of fromObj) {
                    const n = Number(v);
                    if (Number.isInteger(n) && n >= 0 && n <= 3) return n;
                }

                const text = (document.body?.innerText || "").replace(/\\s+/g, " ");
                const patterns = [
                    /(?:GYVYB|GYVYBE|LIVES?)\\D{0,12}([0-3])\\s*\\/\\s*3/i,
                    /(^|\\D)([0-3])\\s*\\/\\s*3(\\D|$)/,
                ];
                for (const p of patterns) {
                    const m = text.match(p);
                    if (!m) continue;
                    const n = Number(m[1] ?? m[2]);
                    if (Number.isInteger(n) && n >= 0 && n <= 3) return n;
                }

                // Last-resort heuristic for UI that uses heart icons only.
                const iconSelector = [
                    "[class*='heart' i]",
                    "[class*='life' i]",
                    "[class*='gyvyb' i]",
                    "img[src*='heart' i]",
                    "img[src*='life' i]",
                    "img[src*='gyvyb' i]",
                ].join(",");
                const visibleIcons = Array.from(document.querySelectorAll(iconSelector))
                    .filter((el) => {
                        const style = window.getComputedStyle(el);
                        return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
                    })
                    .length;
                if (visibleIcons >= 1 && visibleIcons <= 3) return visibleIcons;

                return null;
            })(),
            lifeText: document.getElementById("currentLife")?.textContent?.trim() ?? null,
            resultsVisible: (() => {
                const isVisible = (el) => {
                    if (!el) return false;
                    const style = window.getComputedStyle(el);
                    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") {
                        return false;
                    }
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                };
                const scoreboard = document.getElementById("boomio-competition-scoreboard-name");
                const scoreboardText = scoreboard?.textContent?.toUpperCase() || "";
                if (scoreboardText.includes("REZULTATAI") && isVisible(scoreboard)) return true;
                // Fallback: visible element that itself contains REZULTATAI.
                const rezNodes = Array.from(document.querySelectorAll("*"))
                    .filter((el) => (el.textContent || "").toUpperCase().includes("REZULTATAI"));
                return rezNodes.some((el) => isVisible(el));
            })(),
            loginVisible: !!document.getElementById("input-register-container"),
            // Example of values you may expose from in-page game objects:
            // score: window.game?.score ?? null,
            ts: performance.now()
        })"""
    )


def _all_frames(page):
    # Forms/popups may be rendered inside iframes.
    return [page.main_frame, *page.frames]


def click_first(page, selectors: list[str], timeout_ms: int = 3_000) -> str | None:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for frame in _all_frames(page):
            for selector in selectors:
                loc = frame.locator(selector).first
                if loc.count() > 0:
                    try:
                        loc.click(timeout=750)
                        return selector
                    except PlaywrightTimeoutError:
                        continue
        time.sleep(0.2)
    return None


def fill_first(page, selectors: list[str], value: str, field_name: str, timeout_ms: int = 30_000) -> str:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for frame in _all_frames(page):
            for selector in selectors:
                loc = frame.locator(selector).first
                if loc.count() > 0:
                    try:
                        loc.click(timeout=750)
                        loc.fill(value, timeout=750)
                        return selector
                    except PlaywrightTimeoutError:
                        continue
        time.sleep(0.2)
    raise RuntimeError(
        f"Could not find {field_name} input within {timeout_ms}ms. Tried selectors: {selectors}"
    )


def click_when_stable(locator, stable_ms: int = 1000, timeout_ms: int = 4000) -> bool:
    deadline = time.time() + (timeout_ms / 1000)
    stable_since: float | None = None
    last_box: tuple[float, float, float, float] | None = None

    while time.time() < deadline:
        try:
            if locator.count() == 0:
                time.sleep(0.1)
                continue
            box = locator.bounding_box(timeout=300)
        except PlaywrightTimeoutError:
            time.sleep(0.1)
            continue

        if not box:
            time.sleep(0.1)
            continue

        current = (
            round(box["x"], 1),
            round(box["y"], 1),
            round(box["width"], 1),
            round(box["height"], 1),
        )
        if current == last_box:
            if stable_since is None:
                stable_since = time.time()
            if (time.time() - stable_since) * 1000 >= stable_ms:
                try:
                    locator.click(timeout=500)
                    return True
                except PlaywrightTimeoutError:
                    try:
                        locator.click(timeout=500, force=True)
                        return True
                    except PlaywrightTimeoutError:
                        pass
        else:
            stable_since = None
            last_box = current

        time.sleep(0.1)
    return False


def ensure_privacy_consent(page, timeout_ms: int = 10_000) -> None:
    deadline = time.time() + (timeout_ms / 1000)
    policy_text = "Sutinku, kad mano asmens duomenys būtų tvarkomi tiesioginės rinkodaros tikslu"
    while time.time() < deadline:
        for frame in _all_frames(page):
            # Primary strategy from user report: the correct one is the second checkbox option.
            checked_boxes = frame.locator(".checkbox-checked:visible")
            if checked_boxes.count() >= 2:
                return

            unchecked_boxes = frame.locator(".checkbox-unchecked:visible")
            if unchecked_boxes.count() >= 2:
                if click_when_stable(unchecked_boxes.nth(1), stable_ms=1000, timeout_ms=3500):
                    continue

            # Fallback strategy: find by exact consent text and click its related checkbox row.
            text_block = frame.locator(f"div:has-text('{policy_text}')").first
            if text_block.count() > 0:
                # Scope checkbox lookup to the specific consent row so we don't hit other checkboxes.
                consent_row = text_block.locator("xpath=..")
                if consent_row.locator(".checkbox-checked:visible").count() > 0:
                    return

                unchecked = consent_row.locator(".checkbox-unchecked:visible").first
                if unchecked.count() > 0:
                    if click_when_stable(unchecked, stable_ms=1000, timeout_ms=3500):
                        continue

                try:
                    text_block.click(timeout=750)
                except PlaywrightTimeoutError:
                    continue
        time.sleep(0.2)

    raise RuntimeError("Could not check privacy consent checkbox")


def click_toliau(page, timeout_ms: int = 10_000) -> bool:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for frame in _all_frames(page):
            tol = frame.get_by_text("TOLIAU", exact=True).first
            if tol.count() == 0:
                continue
            candidates = [
                tol,
                tol.locator("xpath=.."),
                tol.locator("xpath=../.."),
                tol.locator("xpath=../../.."),
            ]
            for candidate in candidates:
                try:
                    candidate.click(timeout=750)
                    return True
                except PlaywrightTimeoutError:
                    try:
                        candidate.click(timeout=750, force=True)
                        return True
                    except PlaywrightTimeoutError:
                        continue
        time.sleep(0.2)
    return False


def click_sutinku(page, timeout_ms: int = 15_000) -> bool:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        for frame in _all_frames(page):
            agree = frame.get_by_text("SUTINKU", exact=True).first
            if agree.count() == 0:
                continue
            candidates = [
                agree,
                agree.locator("xpath=.."),
                agree.locator("xpath=../.."),
                agree.locator("xpath=../../.."),
            ]
            for candidate in candidates:
                try:
                    candidate.click(timeout=750)
                    return True
                except PlaywrightTimeoutError:
                    try:
                        candidate.click(timeout=750, force=True)
                        return True
                    except PlaywrightTimeoutError:
                        continue
        time.sleep(0.2)
    return False


def is_game_over_visible(page) -> bool:
    scoreboard = page.locator("#boomio-competition-scoreboard-name").first
    if scoreboard.count() > 0:
        try:
            if scoreboard.is_visible(timeout=250):
                text = (scoreboard.text_content() or "").upper()
                if "REZULTATAI" in text:
                    return True
        except PlaywrightTimeoutError:
            pass

    patterns = [
        re.compile("REZULTATAI", re.IGNORECASE),
        re.compile("GAME OVER", re.IGNORECASE),
        re.compile("PRALAIM", re.IGNORECASE),
        re.compile("BANDYK DAR KART", re.IGNORECASE),
        re.compile("TRY AGAIN", re.IGNORECASE),
    ]
    for frame in _all_frames(page):
        for pattern in patterns:
            loc = frame.get_by_text(pattern).first
            if loc.count() == 0:
                continue
            try:
                if loc.is_visible(timeout=250):
                    return True
            except PlaywrightTimeoutError:
                continue
    return False


def restart_round_with_space(page, timeout_ms: int = 30_000, interval_s: float = 0.5) -> bool:
    deadline = time.time() + (timeout_ms / 1000)
    while time.time() < deadline:
        page.keyboard.press("Space")
        time.sleep(interval_s)
        state = read_state(page)
        lives = state.get("livesRemaining")
        if isinstance(lives, int) and lives == 3:
            return True
    return False


def login(page, email: str) -> None:
    print("[login] waiting for email form")

    email_selector = fill_first(
        page,
        [
            "input[type='email']",
            "input[name*='email' i]",
            "input[id*='email' i]",
            "input[autocomplete='username']",
        ],
        email,
        "email",
        timeout_ms=30_000,
    )
    print(f"[login] filled email field ({email_selector})")
    ensure_privacy_consent(page, timeout_ms=10_000)
    print("[login] privacy consent checked")

    if click_toliau(page, timeout_ms=10_000):
        print("[login] submitted with: TOLIAU")
    else:
        submit_selector = click_first(
            page,
            [
                "button[type='submit']",
                "button:has-text('TOLIAU')",
                "button:has-text('Prisijungti')",
                "button:has-text('Login')",
                "button:has-text('Sign in')",
                "input[type='submit']",
            ],
            timeout_ms=10_000,
        )
        if submit_selector is None:
            raise RuntimeError("Could not find login submit control (including TOLIAU)")
        print(f"[login] submitted with: {submit_selector}")

    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except PlaywrightTimeoutError:
        # Some login flows keep sockets open, so networkidle might not happen.
        pass

    if click_sutinku(page, timeout_ms=15_000):
        print("[login] confirmed with: SUTINKU")
    else:
        raise RuntimeError("Could not find post-login start control: SUTINKU")

    print("[login] login step completed")


def worker(
    worker_id: int,
    out_q: Queue,
    browser_name: str,
    headless: bool,
    steps: int,
    slow_mo_ms: int,
    weights_file: str,
    email: str,
    skip_login: bool,
) -> None:
    with sync_playwright() as p:
        policy = PixelPolicyNet(weights_file=weights_file, verbose=False)
        browser_type = getattr(p, browser_name)
        browser = browser_type.launch(headless=headless, slow_mo=slow_mo_ms)
        page = browser.new_page()
        print(f"[worker {worker_id}] started (browser={browser_name}, headless={headless}, steps={steps})")

        # Capture WebSocket frames; often this is best for precise state/reward.
        def on_websocket(ws):
            ws.on(
                "framereceived",
                lambda frame: out_q.put(
                    {
                        "worker_id": worker_id,
                        "type": "ws_frame",
                        "payload": frame.get("payload"),
                    }
                ),
            )

        page.on("websocket", on_websocket)
        page.goto(GAME_URL, wait_until="domcontentloaded")
        if skip_login:
            print("[worker] skipping login (--skip-login)")
        else:
            login(page, email)

        # Play until game over by default; --steps > 0 can still cap run length.
        total_step = 0
        round_index = 1
        last_lives: int | None = None
        last_score: int | None = None
        round_max_score = 0
        round_started_at = time.time()
        saw_alive_this_round = False
        saved_grid_debug = False
        pending_samples: list[dict[str, Any]] = []

        print(f"[worker {worker_id}] round {round_index} started")
        while True:
            if total_step % 200 == 0:
                policy.reload_if_updated()
            state_before = read_state(page)
            frame_png = capture_game_frame(page)
            x = extract_grid_rgb_features(frame_png)
            action_idx, action, probs, h = policy.sample_action(x)
            key = MOVE_TO_KEY[action]
            if key is not None:
                page.keyboard.press(key)
            time.sleep(0.03)

            state_after = read_state(page)
            score = state_after.get("score")
            lives_remaining = state_after.get("livesRemaining")
            score_before = state_before.get("score")
            lives_before = state_before.get("livesRemaining")
            round_elapsed = time.time() - round_started_at
            done = False
            if isinstance(score, int) and score != last_score:
                last_score = score
                round_max_score = max(round_max_score, score)
            if isinstance(lives_remaining, int):
                if lives_remaining > 0:
                    saw_alive_this_round = True
                if saw_alive_this_round and round_elapsed >= 5 and lives_remaining == 0:
                    done = True
                if lives_remaining != last_lives:
                    print(f"[worker {worker_id}] lives remaining: {lives_remaining}/3")
                    last_lives = lives_remaining

            pending_samples.append(
                {
                    "x": x,
                    "h": h,
                    "probs": probs,
                    "action_idx": action_idx,
                    "reward": 0.0,
                }
            )
            if isinstance(score_before, int) and isinstance(score, int):
                score_delta = score - score_before
                if score_delta >= 50:
                    chunks = score_delta // 50
                    bonus = float(chunks)
                    history = min(3, len(pending_samples) - 1)
                    for i in range(1, history + 1):
                        pending_samples[-1 - i]["reward"] += bonus
            if isinstance(lives_before, int) and isinstance(lives_remaining, int) and lives_remaining < lives_before:
                pending_samples[-1]["reward"] -= float(lives_before - lives_remaining) * 2.0

            reward = float(pending_samples[-1]["reward"])
            if len(pending_samples) > 3:
                sample = pending_samples.pop(0)
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "type": "train_sample",
                        "x": sample["x"].tolist(),
                        "h": sample["h"].tolist(),
                        "probs": sample["probs"].tolist(),
                        "action_idx": int(sample["action_idx"]),
                        "reward": float(sample["reward"]),
                    }
                )

            out_q.put(
                {
                    "worker_id": worker_id,
                    "type": "transition",
                    "step": total_step,
                    "state_before": state_before,
                    "action": action,
                    "key": key,
                    "reward": reward,
                    "state_after": state_after,
                    "score": score,
                    "lives_remaining": lives_remaining,
                    "policy_probs": probs.tolist(),
                    "done": done,
                }
            )
            if (total_step + 1) % 25 == 0:
                score_text = str(score) if isinstance(score, int) else "unknown"
                lives_text = f"{lives_remaining}/3" if isinstance(lives_remaining, int) else "unknown"
                if steps > 0:
                    print(
                        f"[worker {worker_id}] step {total_step + 1}/{steps}, "
                        f"score={score_text}, lives={lives_text}"
                    )
                else:
                    print(f"[worker {worker_id}] step {total_step + 1}, score={score_text}, lives={lives_text}")

            if not saved_grid_debug and (total_step + 1) == 20:
                save_grid_debug_snapshot(page=page, worker_id=worker_id, step=total_step + 1)
                saved_grid_debug = True

            total_step += 1
            if done:
                while pending_samples:
                    sample = pending_samples.pop(0)
                    out_q.put(
                        {
                            "worker_id": worker_id,
                            "type": "train_sample",
                            "x": sample["x"].tolist(),
                            "h": sample["h"].tolist(),
                            "probs": sample["probs"].tolist(),
                            "action_idx": int(sample["action_idx"]),
                            "reward": float(sample["reward"]),
                        }
                    )
                print(f"[worker {worker_id}] game over at step {total_step}")
                out_q.put(
                    {
                        "worker_id": worker_id,
                        "type": "game_over",
                        "round_index": round_index,
                        "round_max_score": round_max_score,
                    }
                )
                time.sleep(1.0)
                if restart_round_with_space(page, timeout_ms=30_000, interval_s=0.5):
                    round_index += 1
                    round_started_at = time.time()
                    saw_alive_this_round = False
                    last_lives = None
                    last_score = None
                    round_max_score = 0
                    policy.reload_if_updated()
                    print(f"[worker {worker_id}] restarted with Space, round {round_index} started")
                    continue
                print("[worker] game over detected but failed to restart with Space")
                break

            if steps > 0 and total_step >= steps:
                while pending_samples:
                    sample = pending_samples.pop(0)
                    out_q.put(
                        {
                            "worker_id": worker_id,
                            "type": "train_sample",
                            "x": sample["x"].tolist(),
                            "h": sample["h"].tolist(),
                            "probs": sample["probs"].tolist(),
                            "action_idx": int(sample["action_idx"]),
                            "reward": float(sample["reward"]),
                        }
                    )
                print(f"[worker {worker_id}] reached step limit ({steps}), stopping worker")
                break

        while pending_samples:
            sample = pending_samples.pop(0)
            out_q.put(
                {
                    "worker_id": worker_id,
                    "type": "train_sample",
                    "x": sample["x"].tolist(),
                    "h": sample["h"].tolist(),
                    "probs": sample["probs"].tolist(),
                    "action_idx": int(sample["action_idx"]),
                    "reward": float(sample["reward"]),
                }
            )
        out_q.put({"worker_id": worker_id, "type": "worker_done"})
        browser.close()


def trainer_loop(num_workers: int, out_q: Queue, learning_rate: float, weights_file: str) -> None:
    """
    Stub trainer loop.
    Replace with replay buffer + model update code.
    """
    done_workers = 0
    transitions = 0
    policy = PixelPolicyNet(weights_file=weights_file, verbose=True)
    train_updates = 0
    reward_ema = 0.0
    grad_norm_ema = 0.0
    max_abs_reward = 0.0
    best_round_score = 0

    while done_workers < num_workers:
        try:
            item = out_q.get(timeout=2)
        except Empty:
            continue

        if item["type"] == "transition":
            transitions += 1
            if transitions % 50 == 0:
                print(f"[trainer] transitions={transitions}")
        elif item["type"] == "train_sample":
            x = np.asarray(item["x"], dtype=np.float32)
            h = np.asarray(item["h"], dtype=np.float32)
            probs = np.asarray(item["probs"], dtype=np.float32)
            metrics = policy.train_step(
                x=x,
                h=h,
                probs=probs,
                action_idx=int(item["action_idx"]),
                reward=float(item["reward"]),
                lr=learning_rate,
            )
            reward = float(item["reward"])
            reward_ema = 0.995 * reward_ema + 0.005 * reward
            grad_norm_ema = 0.995 * grad_norm_ema + 0.005 * metrics["grad_norm"]
            max_abs_reward = max(max_abs_reward, abs(reward))
            train_updates += 1
            if train_updates % 500 == 0:
                policy.save()
            if train_updates % 200 == 0:
                print(
                    "[trainer][metrics] "
                    f"updates={train_updates} reward_ema={reward_ema:.4f} "
                    f"max_abs_reward={max_abs_reward:.2f} grad_norm_ema={grad_norm_ema:.4f} "
                    f"weight_norm={metrics['weight_norm']:.4f} baseline={metrics['baseline']:.4f} "
                    f"best_round_score={best_round_score}"
                )
        elif item["type"] == "game_over":
            policy.save()
            round_score = int(item.get("round_max_score", 0))
            best_round_score = max(best_round_score, round_score)
            print(
                f"[trainer] saved weights on game over (updates={train_updates}, "
                f"worker={item['worker_id']}, round={item.get('round_index')}, "
                f"round_max_score={round_score}, best_round_score={best_round_score})"
            )
        elif item["type"] == "ws_frame":
            payload = item.get("payload")
            if isinstance(payload, str) and len(payload) < 200:
                print(f"[ws][worker {item['worker_id']}] {payload}")
        elif item["type"] == "worker_done":
            done_workers += 1
            print(f"[trainer] worker {item['worker_id']} finished ({done_workers}/{num_workers})")
        else:
            print(f"[trainer] unknown event: {json.dumps(item)[:200]}")

    policy.save()
    print(f"[trainer] all workers done, total transitions={transitions}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Boomio bot example with optional visible browser.")
    parser.add_argument("--url", default=GAME_URL, help="Game URL to open.")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument(
        "--browser",
        choices=["firefox", "chromium", "webkit"],
        default="firefox",
        help="Playwright browser engine to use (default: firefox).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS_PER_WORKER,
        help="Steps per worker. Use 0 to run until game over (default).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default is visible browser).",
    )
    parser.add_argument(
        "--slow-mo-ms",
        type=int,
        default=40,
        help="Delay between Playwright actions in milliseconds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Online training learning rate for the policy network.",
    )
    parser.add_argument(
        "--weights-file",
        default=DEFAULT_WEIGHTS_FILE,
        help="Path to .npz file for loading/saving policy weights.",
    )
    parser.add_argument(
        "--email",
        default=os.getenv("BOOMIO_EMAIL"),
        help="Login email (defaults to BOOMIO_EMAIL env var).",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip login step and start interacting immediately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global GAME_URL
    GAME_URL = args.url
    if not args.skip_login and not args.email:
        raise RuntimeError(
            "Login requires email. Provide --email or set BOOMIO_EMAIL environment variable."
        )

    out_q: Queue = Queue(maxsize=10_000)
    if not os.path.exists(args.weights_file):
        PixelPolicyNet(args.weights_file, verbose=False).save()
    workers = [
        Process(
            target=worker,
            args=(
                i,
                out_q,
                args.browser,
                args.headless,
                args.steps,
                args.slow_mo_ms,
                args.weights_file,
                args.email,
                args.skip_login,
            ),
            daemon=True,
        )
        for i in range(args.workers)
    ]

    run_mode = "until game over" if args.steps == 0 else f"{args.steps} steps"
    print(
        f"[main] starting {args.workers} worker(s), url={GAME_URL}, browser={args.browser}, "
        f"headless={args.headless}, mode={run_mode}"
    )

    for proc in workers:
        proc.start()

    trainer_loop(args.workers, out_q, args.learning_rate, args.weights_file)

    for proc in workers:
        proc.join(timeout=1)


if __name__ == "__main__":
    main()
