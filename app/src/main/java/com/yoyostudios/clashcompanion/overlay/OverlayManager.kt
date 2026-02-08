package com.yoyostudios.clashcompanion.overlay

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Typeface
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import com.yoyostudios.clashcompanion.accessibility.ClashCompanionAccessibilityService
import com.yoyostudios.clashcompanion.capture.ScreenCaptureService
import com.yoyostudios.clashcompanion.command.CommandRouter
import com.yoyostudios.clashcompanion.detection.HandDetector
import com.yoyostudios.clashcompanion.speech.SpeechService
import com.yoyostudios.clashcompanion.util.Coordinates

class OverlayManager(private val context: Context) {

    companion object {
        private const val TAG = "ClashCompanion"
    }

    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
    private var overlayView: View? = null
    private var statusText: TextView? = null
    private var handText: TextView? = null
    private var layoutParams: WindowManager.LayoutParams? = null
    private val handler = Handler(Looper.getMainLooper())

    // Minimize/maximize state
    private var minimizedView: View? = null
    private var minimizedParams: WindowManager.LayoutParams? = null
    private var isMinimized = false

    // Periodic queue check — ensures retries happen even when hand state is stable
    // (e.g., all cards dimmed due to low elixir, no hand changes firing)
    private val queueCheckRunnable = object : Runnable {
        override fun run() {
            CommandRouter.checkQueue()
            handler.postDelayed(this, 2000)
        }
    }

    /**
     * Make overlay transparent to ALL touches (including dispatchGesture).
     * Call before injecting taps so the overlay doesn't intercept them.
     */
    fun setPassthrough(enabled: Boolean) {
        val params = layoutParams ?: return
        val view = overlayView ?: return
        if (enabled) {
            params.flags = params.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        } else {
            params.flags = params.flags and WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE.inv()
        }
        try {
            windowManager.updateViewLayout(view, params)
        } catch (e: Exception) {
            Log.w(TAG, "Failed to update overlay passthrough: ${e.message}")
        }
    }

    fun show() {
        if (overlayView != null) return

        val layout = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(200, 30, 30, 30))
            setPadding(24, 16, 24, 16)
        }

        // Status text
        val status = TextView(context).apply {
            text = "Clash Companion Ready"
            setTextColor(Color.WHITE)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
        }
        statusText = status
        layout.addView(status)

        // Test tap card slot 1
        val btnTapSlot = Button(context).apply {
            text = "Tap Card Slot 1"
            textSize = 12f
            setOnClickListener {
                val svc = ClashCompanionAccessibilityService.instance
                if (svc == null) {
                    updateStatus("ERROR: Accessibility not enabled")
                    return@setOnClickListener
                }
                val (x, y) = Coordinates.CARD_SLOT_1
                val ok = svc.safeTap(x, y)
                updateStatus(if (ok) "Tapped slot 1 ($x, $y)" else "BLOCKED — release screen")
            }
        }
        layout.addView(btnTapSlot)

        // Test play card: slot 1 -> left bridge
        val btnPlayCard = Button(context).apply {
            text = "Play Slot 1 -> Left Bridge"
            textSize = 12f
            setOnClickListener {
                val svc = ClashCompanionAccessibilityService.instance
                if (svc == null) {
                    updateStatus("ERROR: Accessibility not enabled")
                    return@setOnClickListener
                }
                val (sx, sy) = Coordinates.CARD_SLOT_1
                val (zx, zy) = Coordinates.LEFT_BRIDGE
                // Enable passthrough so zone tap doesn't hit the overlay
                setPassthrough(true)
                svc.playCard(sx, sy, zx, zy)
                updateStatus("Playing slot 1 -> left bridge")
                // Restore touchability after gesture completes
                handler.postDelayed({ setPassthrough(false) }, 500)
            }
        }
        layout.addView(btnPlayCard)

        // Test play card: slot 2 -> right bridge
        val btnPlayCard2 = Button(context).apply {
            text = "Play Slot 2 -> Right Bridge"
            textSize = 12f
            setOnClickListener {
                val svc = ClashCompanionAccessibilityService.instance
                if (svc == null) {
                    updateStatus("ERROR: Accessibility not enabled")
                    return@setOnClickListener
                }
                val (sx, sy) = Coordinates.CARD_SLOT_2
                val (zx, zy) = Coordinates.RIGHT_BRIDGE
                setPassthrough(true)
                svc.playCard(sx, sy, zx, zy)
                updateStatus("Playing slot 2 -> right bridge")
                handler.postDelayed({ setPassthrough(false) }, 500)
            }
        }
        layout.addView(btnPlayCard2)

        // Start/Stop Listening toggle
        var isListening = false
        val btnListen = Button(context).apply {
            text = "Start Listening"
            textSize = 12f
            setOnClickListener {
                val svc = SpeechService.instance
                if (svc == null) {
                    updateStatus("ERROR: Start Speech Service first")
                    return@setOnClickListener
                }
                if (!isListening) {
                    svc.onTranscript = { text, latencyMs ->
                        CommandRouter.handleTranscript(text, latencyMs)
                    }
                    svc.startListening()
                    text = "Stop Listening"
                    isListening = true
                    updateStatus("Listening...")
                } else {
                    svc.stopListening()
                    text = "Start Listening"
                    isListening = false
                    updateStatus("Stopped listening")
                }
            }
        }
        layout.addView(btnListen)

        // Hand state display (green text showing detected cards)
        val handDisplay = TextView(context).apply {
            text = ""
            setTextColor(Color.argb(255, 100, 255, 100))
            textSize = 11f
        }
        handText = handDisplay
        layout.addView(handDisplay)

        // Save screenshot test button
        val btnScreenshot = Button(context).apply {
            text = "Save Screenshot"
            textSize = 12f
            setOnClickListener {
                val frame = ScreenCaptureService.getLatestFrame()
                if (frame == null) {
                    updateStatus("ERROR: No frame (start capture first)")
                    return@setOnClickListener
                }
                val saved = saveScreenshot(frame)
                if (saved) {
                    updateStatus("Saved! ${frame.width}x${frame.height}")
                } else {
                    updateStatus("ERROR: Failed to save screenshot")
                }
            }
        }
        layout.addView(btnScreenshot)

        // Minimize button
        val btnMinimize = Button(context).apply {
            text = "Minimize"
            textSize = 12f
            setOnClickListener { minimize() }
        }
        layout.addView(btnMinimize)

        // Close overlay button
        val btnClose = Button(context).apply {
            text = "Close Overlay"
            textSize = 12f
            setOnClickListener { hide() }
        }
        layout.addView(btnClose)

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 20
            y = 200
        }

        layoutParams = params
        overlayView = layout
        windowManager.addView(layout, params)
        CommandRouter.overlay = this
        Log.i(TAG, "Overlay shown")

        // Auto-start hand scanning with ResNet classifier
        if (!HandDetector.isScanning && ScreenCaptureService.instance != null) {
            HandDetector.startScanning(CommandRouter.deckCards, context) { hand ->
                handler.post {
                    updateHandDisplay(hand)
                    CommandRouter.checkQueue()
                }
            }
            updateStatus("Scanning hand (ResNet classifier)")
        }

        // Start periodic queue check (every 2s) for elixir retry
        handler.removeCallbacks(queueCheckRunnable)
        handler.postDelayed(queueCheckRunnable, 2000)
    }

    /**
     * Minimize: hide the full overlay, show a small floating button.
     * Hand scanning and listening KEEP RUNNING — only the UI collapses.
     */
    private fun minimize() {
        if (isMinimized) return

        // Hide the full overlay (but don't remove — keep references alive)
        overlayView?.visibility = View.GONE

        // Create the small floating restore button
        val restoreBtn = Button(context).apply {
            text = "CC"
            textSize = 11f
            setBackgroundColor(Color.argb(180, 30, 30, 30))
            setTextColor(Color.argb(255, 245, 166, 35)) // Gold
            setPadding(16, 8, 16, 8)
            minWidth = 0
            minimumWidth = 0
            minHeight = 0
            minimumHeight = 0
            setOnClickListener { maximize() }
        }

        val miniParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 0
            y = 100
        }

        minimizedView = restoreBtn
        minimizedParams = miniParams
        windowManager.addView(restoreBtn, miniParams)
        isMinimized = true
        Log.i(TAG, "Overlay minimized")
    }

    /**
     * Maximize: restore the full overlay, remove the floating button.
     */
    private fun maximize() {
        if (!isMinimized) return

        // Remove the small button
        minimizedView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
        }
        minimizedView = null
        minimizedParams = null

        // Show the full overlay
        overlayView?.visibility = View.VISIBLE
        isMinimized = false
        Log.i(TAG, "Overlay maximized")
    }

    fun hide() {
        handler.removeCallbacks(queueCheckRunnable)
        HandDetector.stopScanning()
        CommandRouter.clearQueue()
        // Remove minimized button if present
        minimizedView?.let {
            try { windowManager.removeView(it) } catch (_: Exception) {}
            minimizedView = null
            minimizedParams = null
        }
        isMinimized = false
        overlayView?.let {
            windowManager.removeView(it)
            overlayView = null
            statusText = null
            handText = null
            layoutParams = null
            Log.i(TAG, "Overlay hidden")
        }
    }

    fun updateStatus(message: String) {
        statusText?.text = message
        Log.i(TAG, "Status: $message")
    }

    fun updateHandDisplay(hand: Map<Int, String>) {
        val slots = (0..3).map { hand[it] ?: "?" }
        val next = hand[4] ?: "?"
        val queueInfo = CommandRouter.getQueueDisplay()
        val text = "Hand: ${slots.joinToString(" | ")} | Next: $next"
        handText?.text = if (queueInfo.isNotEmpty()) "$text\nQueue:\n$queueInfo" else text
    }

    private fun saveScreenshot(bitmap: Bitmap): Boolean {
        return try {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "clash_test_${System.currentTimeMillis()}.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ClashCompanion")
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
                }
                Log.i(TAG, "Screenshot saved: ${bitmap.width}x${bitmap.height}")
                true
            } else {
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save screenshot: ${e.message}")
            false
        }
    }
}
