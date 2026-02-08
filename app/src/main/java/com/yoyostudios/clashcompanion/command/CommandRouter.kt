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
 * Currently executes:
 *  - FAST path: card slot tap + zone tap via AccessibilityService
 *
 * Stubs (display-only, implemented in future milestones):
 *  - QUEUE (M7), TARGETING (M9), CONDITIONAL (M9), SMART (M8)
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

    /**
     * Main entry point. Called by SpeechService.onTranscript via OverlayManager.
     * Parses the transcript and routes to the appropriate execution tier.
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

        // Parse the transcript
        val cmd = CommandParser.parse(transcript, deckCards)

        val parseMs = System.currentTimeMillis() - parseStart

        Log.i(TAG, "CMD: tier=${cmd.tier} card='${cmd.card}' zone=${cmd.zone} " +
                "conf=${String.format("%.2f", cmd.confidence)} " +
                "parse=${parseMs}ms stt=${sttLatencyMs}ms raw='$transcript'")

        // Route by tier
        when (cmd.tier) {
            CommandTier.IGNORE -> {
                // Silently ignore garbage — don't clutter overlay
                Log.d(TAG, "CMD: Ignored '$transcript'")
            }

            CommandTier.FAST -> {
                executeFastPath(cmd, parseMs, sttLatencyMs)
            }

            CommandTier.QUEUE -> {
                // Stub: display only (M7)
                val msg = if (cmd.card.isNotEmpty()) {
                    "QUEUE: ${cmd.card} -> ${cmd.zone ?: "?"} (M7)"
                } else {
                    "QUEUE: couldn't parse (M7)"
                }
                overlay?.updateStatus(msg)
                Log.i(TAG, "CMD: $msg")
            }

            CommandTier.TARGETING -> {
                // Stub: display only (M9)
                val msg = if (cmd.card.isNotEmpty() && cmd.targetCard != null) {
                    "TARGET: ${cmd.card} the ${cmd.targetCard} (M9)"
                } else {
                    "TARGET: couldn't parse (M9)"
                }
                overlay?.updateStatus(msg)
                Log.i(TAG, "CMD: $msg")
            }

            CommandTier.CONDITIONAL -> {
                // Stub: display only (M9)
                val msg = if (cmd.triggerCard != null && cmd.card.isNotEmpty()) {
                    "RULE: if ${cmd.triggerCard} -> ${cmd.card} ${cmd.zone ?: ""} (M9)"
                } else {
                    "RULE: couldn't parse (M9)"
                }
                overlay?.updateStatus(msg)
                Log.i(TAG, "CMD: $msg")
            }

            CommandTier.SMART -> {
                // Stub: display only (M8)
                if (cmd.card.isNotEmpty()) {
                    // FIX-2: 5+ elixir card with no zone — prompt user
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
