package com.yoyostudios.clashcompanion.deck

import android.content.Context
import android.content.res.AssetManager
import android.net.Uri
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.yoyostudios.clashcompanion.command.CommandRouter

/**
 * Manages deck loading from CR share links and card metadata lookup.
 *
 * Flow: CR Share Link → parse card IDs → lookup in bundled card-data.json → update CommandRouter
 */
object DeckManager {

    private const val TAG = "ClashCompanion"
    private const val PREFS_NAME = "clash_companion_deck"
    private const val PREF_DECK_IDS = "deck_card_ids"

    data class CardInfo(
        val id: Long,
        val name: String,
        val elixir: Int,
        val type: String,
        val key: String
    )

    /** Raw JSON card entry from cr-api-data */
    private data class RawCard(
        val id: Long = 0,
        val name: String = "",
        val elixir: Int = 0,
        val type: String = "",
        val key: String = "",
        val rarity: String = "",
        val is_evolved: Boolean = false
    )

    /** Card ID → CardInfo lookup. Loaded once from assets. */
    private var cardDatabase: Map<Long, CardInfo>? = null

    /** Currently loaded deck (8 cards) */
    var currentDeck: List<CardInfo> = emptyList()
        private set

    /**
     * Load the card database from card-data.json in assets.
     * Call once during app init. Idempotent.
     */
    fun loadCardDatabase(assets: AssetManager) {
        if (cardDatabase != null) return

        try {
            val json = assets.open("card-data.json").bufferedReader().use { it.readText() }
            val listType = object : TypeToken<List<RawCard>>() {}.type
            val rawCards: List<RawCard> = Gson().fromJson(json, listType)

            // Filter out evolved variants and build lookup map
            cardDatabase = rawCards
                .filter { !it.is_evolved }
                .associate { raw ->
                    raw.id to CardInfo(
                        id = raw.id,
                        name = raw.name,
                        elixir = raw.elixir,
                        type = raw.type,
                        key = raw.key
                    )
                }

            Log.i(TAG, "DECK: Card database loaded: ${cardDatabase!!.size} cards")
        } catch (e: Exception) {
            Log.e(TAG, "DECK: Failed to load card database: ${e.message}", e)
            cardDatabase = emptyMap()
        }
    }

    /**
     * Parse a deck share URL into a list of CardInfo.
     * URL format: https://link.clashroyale.com/deck/en?deck=26000002;26000001;...
     * Returns null if URL is invalid or cards not found.
     */
    fun parseDeckUrl(url: String): List<CardInfo>? {
        val db = cardDatabase
        if (db == null || db.isEmpty()) {
            Log.e(TAG, "DECK: Card database not loaded")
            return null
        }

        try {
            val uri = Uri.parse(url)
            val deckParam = uri.getQueryParameter("deck")
            if (deckParam.isNullOrBlank()) {
                Log.w(TAG, "DECK: No 'deck' query param in URL: $url")
                return null
            }

            val ids = deckParam.split(";").mapNotNull { it.trim().toLongOrNull() }
            if (ids.size != 8) {
                Log.w(TAG, "DECK: Expected 8 card IDs, got ${ids.size} from: $deckParam")
                // Still try to load what we can
                if (ids.isEmpty()) return null
            }

            val cards = ids.mapNotNull { id ->
                val card = db[id]
                if (card == null) {
                    Log.w(TAG, "DECK: Unknown card ID: $id")
                }
                card
            }

            if (cards.isEmpty()) {
                Log.e(TAG, "DECK: No valid cards found")
                return null
            }

            Log.i(TAG, "DECK: Parsed ${cards.size} cards from URL")
            return cards
        } catch (e: Exception) {
            Log.e(TAG, "DECK: Failed to parse URL: ${e.message}", e)
            return null
        }
    }

    /**
     * Extract a deck URL from shared text and parse it.
     * The text may contain surrounding message text like:
     * "Check out my deck! https://link.clashroyale.com/deck/en?deck=..."
     */
    fun parseDeckFromText(text: String): List<CardInfo>? {
        // CR sends URLs in two formats:
        // 1. https://link.clashroyale.com/deck/en?deck=26000000;26000001;...
        // 2. https://link.clashroyale.com/en?clashroyale://copyDeck?deck=26000000;...
        // Extract deck param from either format using regex on the raw text

        // Try to find deck= parameter directly in the text
        val deckPattern = Regex("""deck=([0-9;]+)""")
        val deckMatch = deckPattern.find(text)
        if (deckMatch != null) {
            val deckParam = deckMatch.groupValues[1]
            Log.i(TAG, "DECK: Extracted deck param: $deckParam")
            return parseDeckParam(deckParam)
        }

        Log.w(TAG, "DECK: No deck param found in text: $text")
        return null
    }

    /**
     * Parse a semicolon-separated deck parameter string into CardInfo list.
     */
    fun parseDeckParam(deckParam: String): List<CardInfo>? {
        val db = cardDatabase
        if (db == null || db.isEmpty()) {
            Log.e(TAG, "DECK: Card database not loaded")
            return null
        }

        val ids = deckParam.split(";").mapNotNull { it.trim().toLongOrNull() }
        if (ids.isEmpty()) {
            Log.w(TAG, "DECK: No valid card IDs in: $deckParam")
            return null
        }

        val cards = ids.mapNotNull { id ->
            val card = db[id]
            if (card == null) Log.w(TAG, "DECK: Unknown card ID: $id")
            card
        }

        if (cards.isEmpty()) {
            Log.e(TAG, "DECK: No valid cards found")
            return null
        }

        Log.i(TAG, "DECK: Parsed ${cards.size} cards")
        return cards
    }

    /**
     * Set the current deck and update all dependent systems.
     */
    fun setDeck(cards: List<CardInfo>, context: Context? = null) {
        currentDeck = cards

        // Update CommandRouter with card names
        CommandRouter.deckCards = cards.map { it.name }

        // Log the loaded deck
        val names = cards.joinToString(", ") { "${it.name} (${it.elixir})" }
        val avgElixir = cards.map { it.elixir }.average()
        Log.i(TAG, "DECK: Loaded: $names | Avg: %.1f elixir".format(avgElixir))

        // Write deck-boosted hotwords for STT (loaded by SpeechService at init)
        context?.let { writeDeckHotwords(it, cards) }

        // Persist deck IDs for next app launch
        context?.let { ctx ->
            try {
                val prefs = ctx.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                val idsString = cards.joinToString(";") { it.id.toString() }
                prefs.edit().putString(PREF_DECK_IDS, idsString).apply()
                Log.i(TAG, "DECK: Saved deck to preferences")
            } catch (e: Exception) {
                Log.w(TAG, "DECK: Failed to save deck: ${e.message}")
            }
        }
    }

    /** Hotwords filename written to filesDir for SpeechService */
    const val DECK_HOTWORDS_FILE = "deck_hotwords.txt"

    /**
     * Write a deck-boosted hotwords file for sherpa-onnx STT.
     * Deck card names get weight 6.0 (strong bias), non-deck cards get 1.0.
     * This dramatically improves STT accuracy for the user's actual deck.
     */
    private fun writeDeckHotwords(context: Context, cards: List<CardInfo>) {
        try {
            val deckNames = cards.map { it.name.uppercase() }.toSet()

            // Read base hotwords from assets
            val baseLines = context.assets.open("hotwords.txt")
                .bufferedReader().use { it.readLines() }

            val sb = StringBuilder()
            for (line in baseLines) {
                val trimmed = line.trim()
                if (trimmed.isBlank()) continue
                val colonIdx = trimmed.lastIndexOf(':')
                if (colonIdx < 0) { sb.appendLine(trimmed); continue }

                val word = trimmed.substring(0, colonIdx).trim()

                // Check if this hotword matches any deck card name
                val isDeckCard = deckNames.any { deckName ->
                    word == deckName || deckName.contains(word) || word.contains(deckName)
                }

                // Deck cards: strong boost (6.0). Non-deck: minimal (1.0)
                val weight = if (isDeckCard) 6.0f else 1.0f
                sb.appendLine("$word :$weight")
            }

            val file = java.io.File(context.filesDir, DECK_HOTWORDS_FILE)
            file.writeText(sb.toString())
            Log.i(TAG, "DECK: Wrote deck-boosted hotwords (${deckNames.size} cards boosted)")
        } catch (e: Exception) {
            Log.w(TAG, "DECK: Failed to write deck hotwords: ${e.message}")
        }
    }

    /**
     * Load previously saved deck from SharedPreferences.
     * Returns true if a deck was loaded.
     */
    fun loadSavedDeck(context: Context): Boolean {
        val db = cardDatabase ?: return false
        try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val idsString = prefs.getString(PREF_DECK_IDS, null) ?: return false

            val ids = idsString.split(";").mapNotNull { it.trim().toLongOrNull() }
            val cards = ids.mapNotNull { db[it] }

            if (cards.size >= 4) { // At least half a deck
                setDeck(cards)
                Log.i(TAG, "DECK: Restored saved deck (${cards.size} cards)")
                return true
            }
        } catch (e: Exception) {
            Log.w(TAG, "DECK: Failed to load saved deck: ${e.message}")
        }
        return false
    }

    /**
     * Get a formatted deck summary string for UI display.
     */
    fun getDeckSummary(): String {
        if (currentDeck.isEmpty()) return "No deck loaded"

        val sb = StringBuilder()
        sb.appendLine("Deck (${currentDeck.size} cards):")
        for (card in currentDeck) {
            sb.appendLine("  ${card.name} — ${card.elixir} elixir (${card.type})")
        }
        val avg = currentDeck.map { it.elixir }.average()
        sb.append("Avg elixir: %.1f".format(avg))
        return sb.toString()
    }

    /**
     * Get the card art image URL from RoyaleAPI CDN.
     */
    fun getCardImageUrl(card: CardInfo): String =
        "https://raw.githubusercontent.com/RoyaleAPI/cr-api-assets/master/cards-150/${card.key}.png"

    /**
     * Build a JSON representation of the deck for sending to Claude.
     */
    fun getDeckJson(): String {
        val cards = currentDeck.map { card ->
            mapOf(
                "name" to card.name,
                "elixir" to card.elixir,
                "type" to card.type
            )
        }
        return Gson().toJson(cards)
    }
}
