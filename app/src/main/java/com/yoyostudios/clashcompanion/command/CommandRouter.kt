package com.yoyostudios.clashcompanion.command

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.yoyostudios.clashcompanion.accessibility.ClashCompanionAccessibilityService
import com.yoyostudios.clashcompanion.command.CommandParser.CommandTier
import com.yoyostudios.clashcompanion.detection.HandDetector
import com.yoyostudios.clashcompanion.overlay.OverlayManager
import com.yoyostudios.clashcompanion.util.Coordinates

/**
 * Routes parsed commands to execution.
 *
 * Executes:
 *  - FAST path: card slot tap + zone tap via AccessibilityService
 *  - QUEUE path: buffered commands that auto-play when card appears in hand
 *
 * Stubs (display-only, implemented in future milestones):
 *  - TARGETING (M9), CONDITIONAL (M9), SMART (M8)
 */
object CommandRouter {

    private const val TAG = "ClashCompanion"

    /** Set by OverlayManager when overlay is shown */
    var overlay: OverlayManager? = null

    /** Current deck card names — set by DeckManager, used by CommandParser */
    var deckCards = listOf(
        "Knight", "Archers", "Minions", "Arrows",
        "Fireball", "Giant", "Musketeer", "Mini P.E.K.K.A"
    )

    /** Prevents rapid-fire double plays during tap animation */
    private var isBusy = false

    private val handler = Handler(Looper.getMainLooper())

    // ── Queue Path (M7) ────────────────────────────────────────────────

    /** A buffered command waiting for a card to appear in hand */
    data class QueuedCommand(
        val card: String,
        val zone: String,
        val timestamp: Long = System.currentTimeMillis(),
        /** Tracks last play attempt for elixir retry. 0 = never attempted. */
        var lastAttemptMs: Long = 0L
    )

    /** FIFO queue of pending commands. Accessed only on main thread. */
    private val commandQueue = mutableListOf<QueuedCommand>()

    /** Queue entries expire after 2 minutes to prevent stale surprise plays */
    private const val QUEUE_TIMEOUT_MS = 120_000L

    /** Cooldown between retry attempts when card is in hand but elixir was low */
    private const val QUEUE_RETRY_COOLDOWN_MS = 1500L

    /** Grace period before removing a "played" entry — card may just be dimmed (low elixir) */
    private const val QUEUE_PLAYED_GRACE_MS = 5000L

    /**
     * Main entry point. Called by SpeechService.onTranscript via OverlayManager.
     * Parses the transcript and routes to the appropriate execution tier.
     *
     * Supports "then"-chaining: "knight left then musketeer right" plays
     * knight immediately and queues musketeer for auto-play when it appears.
     *
     * @param transcript raw STT output text
     * @param sttLatencyMs time taken by STT inference in milliseconds
     */
    fun handleTranscript(transcript: String, sttLatencyMs: Long) {
        // Guard: don't fire while previous tap is animating
        if (isBusy) {
            Log.d(TAG, "CMD: Busy, skipping '$transcript'")
            overlay?.updateStatus("Wait...")
            return
        }

        val parseStart = System.currentTimeMillis()

        // Split on "then"/"than" for chained commands:
        // "knight left then musketeer right" → play knight, queue musketeer
        val cleaned = CommandParser.cleanTranscript(transcript)
        val normalized = cleaned
            .replace(" and then ", " then ")
            .replace(" than ", " then ")
        val thenParts = normalized.split(" then ")

        // Parse and route the primary (first) segment
        val firstSegment = thenParts[0].trim()
        if (firstSegment.isNotEmpty()) {
            val cmd = CommandParser.parse(firstSegment, deckCards)
            val parseMs = System.currentTimeMillis() - parseStart

            Log.i(TAG, "CMD: tier=${cmd.tier} card='${cmd.card}' zone=${cmd.zone} " +
                    "conf=${String.format("%.2f", cmd.confidence)} " +
                    "parse=${parseMs}ms stt=${sttLatencyMs}ms raw='$transcript'")

            // Route by tier
            routeCommand(cmd, parseMs, sttLatencyMs, transcript)
        } else if (thenParts.size <= 1) {
            // Empty transcript with no "then" part — ignore
            Log.d(TAG, "CMD: Empty transcript, ignoring")
        }

        // Queue any "then"-chained segments
        for (i in 1 until thenParts.size) {
            val thenSegment = thenParts[i].trim()
            if (thenSegment.isEmpty()) continue
            queueThenSegment(thenSegment)
        }
    }

    /**
     * Route a parsed command to the correct execution tier.
     */
    private fun routeCommand(
        cmd: CommandParser.ParsedCommand,
        parseMs: Long,
        sttLatencyMs: Long,
        rawTranscript: String
    ) {
        when (cmd.tier) {
            CommandTier.IGNORE -> {
                Log.d(TAG, "CMD: Ignored '$rawTranscript'")
            }

            CommandTier.FAST -> {
                executeFastPath(cmd, parseMs, sttLatencyMs)
            }

            CommandTier.QUEUE -> {
                executeQueuePath(cmd, parseMs, sttLatencyMs)
            }

            CommandTier.TARGETING -> {
                val msg = if (cmd.card.isNotEmpty() && cmd.targetCard != null) {
                    "TARGET: ${cmd.card} the ${cmd.targetCard} (M9)"
                } else {
                    "TARGET: couldn't parse (M9)"
                }
                overlay?.updateStatus(msg)
                Log.i(TAG, "CMD: $msg")
            }

            CommandTier.CONDITIONAL -> {
                val msg = if (cmd.triggerCard != null && cmd.card.isNotEmpty()) {
                    "RULE: if ${cmd.triggerCard} -> ${cmd.card} ${cmd.zone ?: ""} (M9)"
                } else {
                    "RULE: couldn't parse (M9)"
                }
                overlay?.updateStatus(msg)
                Log.i(TAG, "CMD: $msg")
            }

            CommandTier.SMART -> {
                if (cmd.card.isNotEmpty()) {
                    val msg = "Where? Say '${cmd.card} left/right/center'"
                    overlay?.updateStatus(msg)
                    Log.i(TAG, "CMD: $msg")
                } else {
                    val msg = "SMART: '${cmd.rawTranscript}' (M8)"
                    overlay?.updateStatus(msg)
                    Log.i(TAG, "CMD: $msg")
                }
            }
        }
    }

    /**
     * Queue a "then"-chained segment. Always adds to queue (never immediate).
     * The user said "X then Y" — Y should wait, even if Y is in hand right now.
     */
    private fun queueThenSegment(segment: String) {
        val cmd = CommandParser.parse(segment, deckCards)
        if (cmd.card.isEmpty()) {
            Log.w(TAG, "CMD: Then-chain couldn't parse: '$segment'")
            return
        }
        val zone = cmd.zone ?: CommandParser.getDefaultZone(cmd.card)
        if (zone == null) {
            overlay?.updateStatus("Where? Say '${cmd.card} left/right/center'")
            return
        }
        commandQueue.add(QueuedCommand(cmd.card, zone))
        val msg = "THEN: ${cmd.card} -> $zone (queued)"
        overlay?.updateStatus(msg)
        Log.i(TAG, "CMD: $msg (${commandQueue.size} in queue)")
    }

    // ── Queue Path Execution ─────────────────────────────────────────────

    /**
     * Handle a QUEUE-tier command: "queue balloon bridge" / "next hog left" / "then hog left"
     *
     * Always adds to the command queue. If the card is already in hand,
     * triggers checkQueue() immediately for a fast play attempt. If elixir
     * is too low and the tap fails, checkQueue will retry every 1.5s until
     * the card leaves hand (= successfully played).
     */
    private fun executeQueuePath(
        cmd: CommandParser.ParsedCommand,
        parseMs: Long,
        sttLatencyMs: Long
    ) {
        // Validate: parser must have matched a card
        if (cmd.card.isEmpty()) {
            overlay?.updateStatus("Queue: couldn't parse card")
            Log.w(TAG, "CMD: Queue parse failed for '${cmd.rawTranscript}'")
            return
        }

        // Validate: zone must be present (5+ elixir with no zone returns null)
        if (cmd.zone == null) {
            val msg = "Where? Say '${cmd.card} left/right/center'"
            overlay?.updateStatus(msg)
            Log.i(TAG, "CMD: Queue needs zone — $msg")
            return
        }

        // Always add to queue — checkQueue handles execution + elixir retry
        commandQueue.add(QueuedCommand(cmd.card, cmd.zone))
        Log.i(TAG, "CMD: QUEUED: ${cmd.card} -> ${cmd.zone} (${commandQueue.size} in queue)")

        // If card is already in hand, trigger immediate play attempt
        if (!isBusy && HandDetector.cardToSlot.containsKey(cmd.card)) {
            checkQueue()
        } else {
            overlay?.updateStatus("QUEUED: ${cmd.card} -> ${cmd.zone} (waiting...)")
        }
    }

    /**
     * Check if any queued card has appeared in the player's hand.
     * Called from OverlayManager's onHandChanged callback on the main thread.
     *
     * Retry-with-cooldown logic for elixir handling:
     * - When a card is in hand, attempt to play it (tap card slot + zone)
     * - Don't remove entry yet — if elixir was too low, the tap does nothing
     * - Wait 1.5s cooldown before retrying (elixir regeneration time)
     * - Remove entry only when card LEAVES hand (= successfully played)
     *
     * Executes at most one queued command per call (isBusy gates 500ms gap).
     * FIFO iteration order — first eligible queued card wins.
     */
    fun checkQueue() {
        if (isBusy || commandQueue.isEmpty()) return

        val now = System.currentTimeMillis()

        // Prune expired entries (>2 minutes old)
        commandQueue.removeAll { now - it.timestamp > QUEUE_TIMEOUT_MS }
        if (commandQueue.isEmpty()) return

        val handSlots = HandDetector.cardToSlot

        // Remove entries where card has been absent for >5s after attempt.
        // This distinguishes "card truly played" (gone for 5+ seconds) from
        // "card temporarily dimmed" (gone for ~1-2s due to low elixir).
        // Dimmed cards come back when elixir regenerates; played cards don't.
        commandQueue.removeAll { q ->
            q.lastAttemptMs > 0
                    && !handSlots.containsKey(q.card)
                    && now - q.lastAttemptMs > QUEUE_PLAYED_GRACE_MS
        }
        if (commandQueue.isEmpty()) return

        // Find first queued card that's in hand and not cooling down
        for (queued in commandQueue) {
            // Card not in hand yet — skip, keep waiting
            if (!handSlots.containsKey(queued.card)) continue

            // Card is in hand but cooling down from a failed attempt (elixir too low)
            if (queued.lastAttemptMs > 0 && now - queued.lastAttemptMs < QUEUE_RETRY_COOLDOWN_MS) continue

            // Card is in hand and ready to play!
            val isRetry = queued.lastAttemptMs > 0
            queued.lastAttemptMs = now
            val waitMs = now - queued.timestamp

            Log.i(TAG, "CMD: Queue ${if (isRetry) "retry" else "triggered"} — " +
                    "${queued.card} in hand (waited ${waitMs}ms)")

            val msg = "QUEUE: ${queued.card} -> ${queued.zone}" +
                    if (isRetry) " (retry)" else ""
            overlay?.updateStatus(msg)

            // Execute via Fast Path with full safety checks
            executeFastPath(
                CommandParser.ParsedCommand(
                    tier = CommandTier.FAST,
                    card = queued.card,
                    zone = queued.zone,
                    confidence = 1.0f,
                    rawTranscript = "QUEUE auto-play"
                ),
                parseMs = 0,
                sttLatencyMs = waitMs
            )
            return // One at a time — isBusy blocks until 500ms release
        }
    }

    /**
     * Clear all pending queue entries.
     * Called from OverlayManager.hide() to prevent phantom plays.
     */
    fun clearQueue() {
        val count = commandQueue.size
        commandQueue.clear()
        if (count > 0) {
            overlay?.updateStatus("Queue cleared ($count removed)")
            Log.i(TAG, "CMD: Queue cleared ($count entries)")
        }
    }

    /**
     * Get a display string of active queue entries for the overlay.
     * Returns empty string if queue is empty.
     */
    fun getQueueDisplay(): String {
        if (commandQueue.isEmpty()) return ""
        return commandQueue.joinToString("\n") { q ->
            val ago = (System.currentTimeMillis() - q.timestamp) / 1000
            "  ${q.card} -> ${q.zone} (${ago}s)"
        }
    }

    // ── Fast Path Execution ───────────────────────────────────────────────

    /**
     * Execute the Fast Path: tap card slot, then tap zone.
     * Includes confidence gating and hand verification.
     */
    private fun executeFastPath(
        cmd: CommandParser.ParsedCommand,
        parseMs: Long,
        sttLatencyMs: Long
    ) {
        val card = cmd.card
        val zone = cmd.zone

        // Confidence gate
        val threshold = CommandParser.getConfidenceThreshold(card)
        if (cmd.confidence < threshold) {
            val pct = (cmd.confidence * 100).toInt()
            val msg = "Low confidence: $card? ($pct% < ${(threshold * 100).toInt()}%)"
            overlay?.updateStatus(msg)
            Log.w(TAG, "CMD: $msg")
            return
        }

        // Check card is in hand via ResNet classifier
        val slotIndex = HandDetector.cardToSlot[card]
        if (slotIndex == null) {
            // Only show hand slots 0-3, not next-card slot 4
            val handCards = HandDetector.currentHand
                .filterKeys { it < 4 }
                .entries.sortedBy { it.key }
                .joinToString { it.value }
            val msg = "Not in hand: $card (hand: $handCards)"
            overlay?.updateStatus(msg)
            Log.w(TAG, "CMD: $msg")
            return
        }

        // Get coordinates
        if (slotIndex < 0 || slotIndex >= Coordinates.CARD_SLOTS.size) {
            val msg = "ERROR: Invalid slot index $slotIndex for $card"
            overlay?.updateStatus(msg)
            Log.e(TAG, "CMD: $msg")
            return
        }

        val (slotX, slotY) = Coordinates.CARD_SLOTS[slotIndex]

        // Zone coordinates
        if (zone == null) {
            val msg = "ERROR: No zone for $card"
            overlay?.updateStatus(msg)
            Log.e(TAG, "CMD: $msg")
            return
        }

        val zoneCoords = Coordinates.ZONE_MAP[zone]
        if (zoneCoords == null) {
            val msg = "ERROR: Unknown zone '$zone'"
            overlay?.updateStatus(msg)
            Log.e(TAG, "CMD: $msg")
            return
        }

        val (zoneX, zoneY) = zoneCoords

        // Execute!
        val svc = ClashCompanionAccessibilityService.instance
        if (svc == null) {
            val msg = "ERROR: Accessibility service not connected"
            overlay?.updateStatus(msg)
            Log.e(TAG, "CMD: $msg")
            return
        }

        isBusy = true

        val totalMs = parseMs + sttLatencyMs
        val msg = "FAST: $card -> $zone (${parseMs}ms + STT ${sttLatencyMs}ms = ${totalMs}ms)"
        overlay?.updateStatus(msg)
        Log.i(TAG, "CMD: $msg — tapping slot $slotIndex ($slotX, $slotY) -> zone ($zoneX, $zoneY)")

        // Make overlay transparent to touches so dispatchGesture doesn't hit overlay buttons
        overlay?.setPassthrough(true)
        svc.playCard(slotX, slotY, zoneX, zoneY)

        // Release busy lock and restore overlay touchability after gesture completes
        handler.postDelayed({
            overlay?.setPassthrough(false)
            isBusy = false
        }, 500)
    }
}
