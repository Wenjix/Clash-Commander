package com.yoyostudios.clashcompanion.command

import com.yoyostudios.clashcompanion.util.CardAliases
import com.yoyostudios.clashcompanion.util.Coordinates

/**
 * Five-tier command parser. Pure parsing logic — no Android dependencies.
 *
 * Parse flow (corrected with all 7 fixes from cross-reference analysis):
 *  1. cleanTranscript
 *  2. isNonCommand filter
 *  3. COMPOUND_ALIASES check (FIX-1: "nitrate" -> Knight + right_bridge)
 *  4. Split on " then " (FIX-4: process first segment only)
 *  5. Prefix detection: "if " -> CONDITIONAL, "queue "/"next " -> QUEUE
 *  6. "the" disambiguation: zone vs targeting
 *  7. extractZone (longest-match-first, word-boundary)
 *  8. stripFillers (FIX-3: remove "play"/"drop"/"the"/etc)
 *  9. matchCard (FIX-6: full-text + word-level + bigram + fuzzy)
 * 10. Route: card+zone -> FAST, card only >=5 elixir -> reject (FIX-2),
 *     card only <5 -> default zone, no card -> SMART
 */
object CommandParser {

    // ── Data classes ────────────────────────────────────────────────────

    enum class CommandTier { FAST, QUEUE, TARGETING, SMART, CONDITIONAL, IGNORE }

    data class ParsedCommand(
        val tier: CommandTier,
        val card: String = "",
        val zone: String? = null,
        val targetCard: String? = null,
        val triggerCard: String? = null,
        val confidence: Float = 0f,
        val rawTranscript: String = ""
    )

    // ── Public API ──────────────────────────────────────────────────────

    /**
     * Parse a transcript into a routable command.
     * @param transcript raw STT output (lowercase, may have punctuation)
     * @param deckCards list of 8 official card names in the player's deck
     */
    fun parse(transcript: String, deckCards: List<String>): ParsedCommand {
        // Step 1: Clean
        val text = cleanTranscript(transcript)

        // Step 2: Filter garbage
        if (isNonCommand(text)) {
            return ParsedCommand(CommandTier.IGNORE, rawTranscript = transcript)
        }

        // Step 3: Compound alias check (FIX-1)
        val compound = checkCompoundAlias(text)
        if (compound != null) return compound

        // Step 4: Split on "then" — process first segment only (FIX-4)
        val segment = text.split(" then ")[0].trim()
        if (segment.isEmpty()) {
            return ParsedCommand(CommandTier.IGNORE, rawTranscript = transcript)
        }

        // Step 5a: Conditional pattern — "if [trigger] [action]"
        if (segment.startsWith("if ")) {
            return parseConditional(segment.removePrefix("if ").trim(), deckCards, transcript)
        }

        // Step 5b: Queue pattern — "queue/next/then [card] [zone]"
        val queuePrefix = when {
            segment.startsWith("queue ") -> "queue "
            segment.startsWith("next ") -> "next "
            segment.startsWith("then ") -> "then "
            else -> null
        }
        if (queuePrefix != null) {
            return parseQueueCommand(segment.removePrefix(queuePrefix).trim(), deckCards, transcript)
        }

        // Step 6: "the" disambiguation
        if (segment.contains(" the ")) {
            val theResult = handleTheDisambiguation(segment, deckCards, transcript)
            if (theResult != null) return theResult
        }

        // Step 7: Normal flow — extract zone, match card
        return parseNormalCommand(segment, deckCards, transcript)
    }

    /**
     * Confidence threshold scaled by elixir cost (master plan Section 14).
     * Higher cost = higher threshold = fewer accidental plays.
     */
    fun getConfidenceThreshold(cardName: String): Float {
        val elixir = CardAliases.CARD_ELIXIR[cardName] ?: 3
        return when {
            elixir >= 5 -> 0.85f
            elixir >= 3 -> 0.70f
            else -> 0.60f
        }
    }

    // ── Transcript cleaning ─────────────────────────────────────────────

    /**
     * Normalize STT output:
     *  - lowercase
     *  - strip punctuation (Zipformer adds trailing periods)
     *  - collapse spaces
     *  - deduplicate consecutive repeated words (Zipformer short-utterance bug)
     *  - trim
     */
    fun cleanTranscript(raw: String): String {
        var text = raw.lowercase()
            .replace(Regex("[^a-z0-9 ]"), "")
            .replace(Regex("\\s+"), " ")
            .trim()

        // Deduplicate consecutive repeated words:
        // "giant giant giant bridge" -> "giant bridge"
        val words = text.split(" ")
        val deduped = mutableListOf<String>()
        for (word in words) {
            if (deduped.isEmpty() || deduped.last() != word) {
                deduped.add(word)
            }
        }
        return deduped.joinToString(" ")
    }

    /**
     * Returns true if the text is not a game command (filler, garbage, too short).
     */
    fun isNonCommand(text: String): Boolean {
        if (text.length < 3) return true
        if (text in CardAliases.NON_COMMANDS) return true
        // All words are filler — no card reference at all
        val words = text.split(" ")
        if (words.all { it in CardAliases.FILLER_WORDS }) return true
        return false
    }

    // ── Compound alias check (FIX-1) ────────────────────────────────────

    /**
     * Checks if the entire transcript is a known compound misrecognition
     * where STT merges card+zone into one word (e.g. "nitrate" = "knight right").
     * Must be checked BEFORE zone extraction.
     */
    private fun checkCompoundAlias(text: String): ParsedCommand? {
        val match = CardAliases.COMPOUND_ALIASES[text] ?: return null
        return ParsedCommand(
            tier = CommandTier.FAST,
            card = match.first,
            zone = match.second,
            confidence = 1.0f,
            rawTranscript = text
        )
    }

    // ── Zone extraction ─────────────────────────────────────────────────

    /**
     * Extract the zone from text using longest-match-first word-boundary matching.
     * Returns (zone_key, remaining_text_with_zone_removed).
     * If no zone found, returns (null, original_text).
     */
    fun extractZone(text: String): Pair<String?, String> {
        // Sort zone keys by length descending so "left bridge" matches before "left"
        val sortedKeys = Coordinates.ZONE_MAP.keys.sortedByDescending { it.length }

        for (key in sortedKeys) {
            // Word-boundary match to avoid matching "left" inside "leftover"
            val pattern = Regex("\\b${Regex.escape(key)}\\b")
            if (pattern.containsMatchIn(text)) {
                val remaining = text.replace(pattern, "")
                    .replace(Regex("\\s+"), " ")
                    .trim()
                return key to remaining
            }
        }

        return null to text
    }

    // ── Filler word stripping (FIX-3) ───────────────────────────────────

    /**
     * Remove common filler/action words that aren't card names.
     * "play knight" -> "knight", "drop the archers" -> "archers"
     * If stripping removes everything, returns original text.
     */
    fun stripFillers(text: String): String {
        val words = text.split(" ")
        val filtered = words.filter { it !in CardAliases.FILLER_WORDS }
        val result = filtered.joinToString(" ").trim()
        return if (result.isEmpty()) text else result
    }

    // ── Card matching (FIX-6: multi-level) ──────────────────────────────

    /**
     * Multi-level card matching strategy (deck-aware):
     *  1. Full-text alias lookup — prefer deck cards, save non-deck as fallback
     *  2. Word + bigram alias lookup — prefer deck cards, save non-deck as fallback
     *  3. Fuzzy match vs deck card names
     *  4. Fuzzy match vs all alias keys
     *
     * Deck-aware: if an alias maps to a card NOT in the deck, checks whether
     * a related deck card exists (e.g. "pekka" -> P.E.K.K.A not in deck,
     * but Mini P.E.K.K.A is -> returns Mini P.E.K.K.A).
     *
     * FIX-7: Short card names (<=3 chars) require similarity >= 0.75
     * to prevent "tap" matching "Zap".
     */
    fun matchCard(text: String, deckCards: List<String>): Pair<String, Float>? {
        if (text.isBlank()) return null

        val deckSet = deckCards.toSet()
        var nonDeckFallback: Pair<String, Float>? = null

        // Helper: strip dots, hyphens, and other punctuation for substring matching
        fun normalize(s: String): String = s.lowercase().replace(Regex("[^a-z ]"), "").trim()

        // Helper: if aliasCard is in deck, return it directly.
        // If not, check if any deck card's normalized name contains the
        // normalized alias result (handles Mini P.E.K.K.A containing P.E.K.K.A).
        fun findDeckVariant(aliasCard: String): String? {
            if (aliasCard in deckSet) return aliasCard
            val norm = normalize(aliasCard)
            if (norm.isBlank()) return null
            return deckCards.firstOrNull { normalize(it).contains(norm) }
        }

        // Level 1: Full-text direct alias lookup
        CardAliases.CARD_ALIASES[text]?.let { card ->
            val resolved = findDeckVariant(card)
            if (resolved != null) return resolved to 1.0f
            if (nonDeckFallback == null) nonDeckFallback = card to 0.9f
        }

        // Level 2: Word + bigram alias lookup
        val words = text.split(" ").filter { it.isNotBlank() }

        // Try bigrams first (more specific: "hog rider" > "hog")
        if (words.size >= 2) {
            for (i in 0 until words.size - 1) {
                val bigram = "${words[i]} ${words[i + 1]}"
                CardAliases.CARD_ALIASES[bigram]?.let { card ->
                    val resolved = findDeckVariant(card)
                    if (resolved != null) return resolved to 1.0f
                    if (nonDeckFallback == null) nonDeckFallback = card to 0.9f
                }
            }
        }

        // Try individual words
        for (word in words) {
            if (word.length < 2) continue // skip single-char noise
            CardAliases.CARD_ALIASES[word]?.let { card ->
                val resolved = findDeckVariant(card)
                if (resolved != null) return resolved to 1.0f
                if (nonDeckFallback == null) nonDeckFallback = card to 0.9f
            }
        }

        // Level 3: Fuzzy match against deck card names (8 candidates)
        var bestCard: String? = null
        var bestSimilarity = 0f

        for (deckCard in deckCards) {
            val deckLower = deckCard.lowercase()

            // Full text vs deck card name
            val sim = levenshteinSimilarity(text, deckLower)
            if (sim > bestSimilarity) {
                bestSimilarity = sim
                bestCard = deckCard
            }

            // Each word vs deck card name
            for (word in words) {
                if (word.length < 2) continue
                val wordSim = levenshteinSimilarity(word, deckLower)
                if (wordSim > bestSimilarity) {
                    bestSimilarity = wordSim
                    bestCard = deckCard
                }
            }
        }

        // Level 4: Fuzzy match against all alias keys
        // Always resolve through findDeckVariant so non-deck cards
        // get mapped to their deck variant (e.g. P.E.K.K.A -> Mini P.E.K.K.A)
        for ((alias, card) in CardAliases.CARD_ALIASES) {
            val sim = levenshteinSimilarity(text, alias)
            if (sim > bestSimilarity) {
                val resolved = findDeckVariant(card)
                if (resolved != null) {
                    bestSimilarity = sim
                    bestCard = resolved
                } else if (bestCard == null || bestCard !in deckSet) {
                    // Only use non-deck card if we don't have a deck match yet
                    bestSimilarity = sim
                    bestCard = card
                }
            }
            for (word in words) {
                if (word.length < 2) continue
                val wordSim = levenshteinSimilarity(word, alias)
                if (wordSim > bestSimilarity) {
                    val resolved = findDeckVariant(card)
                    if (resolved != null) {
                        bestSimilarity = wordSim
                        bestCard = resolved
                    } else if (bestCard == null || bestCard !in deckSet) {
                        bestSimilarity = wordSim
                        bestCard = card
                    }
                }
            }
        }

        // FIX-7: Short card names need higher threshold
        val minThreshold = if (bestCard != null && bestCard.length <= 3) 0.75f else 0.5f

        if (bestCard != null && bestSimilarity >= minThreshold) {
            return bestCard to bestSimilarity
        }

        // Last resort: return non-deck alias fallback (card exists but not in this deck)
        return nonDeckFallback
    }

    // ── Default zone logic (FIX-2) ──────────────────────────────────────

    /**
     * Returns default zone for a card when no zone is specified.
     * Returns null for 5+ elixir cards (master plan: "REQUIRE explicit location").
     */
    fun getDefaultZone(cardName: String): String? {
        val elixir = CardAliases.CARD_ELIXIR[cardName] ?: 3
        if (elixir >= 5) return null // FIX-2: expensive cards require explicit zone

        return when (CardAliases.CARD_TYPES[cardName]) {
            "building" -> "center"
            "spell" -> "center"
            else -> "left_bridge" // troops default to left bridge
        }
    }

    // ── Levenshtein ─────────────────────────────────────────────────────

    fun levenshteinDistance(a: String, b: String): Int {
        val m = a.length
        val n = b.length
        val dp = Array(m + 1) { IntArray(n + 1) }

        for (i in 0..m) dp[i][0] = i
        for (j in 0..n) dp[0][j] = j

        for (i in 1..m) {
            for (j in 1..n) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                dp[i][j] = minOf(
                    dp[i - 1][j] + 1,      // deletion
                    dp[i][j - 1] + 1,      // insertion
                    dp[i - 1][j - 1] + cost // substitution
                )
            }
        }

        return dp[m][n]
    }

    fun levenshteinSimilarity(a: String, b: String): Float {
        if (a.isEmpty() && b.isEmpty()) return 1.0f
        val maxLen = maxOf(a.length, b.length)
        if (maxLen == 0) return 1.0f
        return 1.0f - (levenshteinDistance(a, b).toFloat() / maxLen)
    }

    // ── "the" disambiguation ────────────────────────────────────────────

    /**
     * Handles commands containing " the ":
     *  - "log the bridge" -> FAST (log at bridge, not targeting)
     *  - "fireball the hog rider" -> TARGETING (fireball on hog rider)
     *
     * Returns null if disambiguation is inconclusive (fall through to normal).
     */
    private fun handleTheDisambiguation(
        segment: String,
        deckCards: List<String>,
        rawTranscript: String
    ): ParsedCommand? {
        val parts = segment.split(" the ", limit = 2)
        if (parts.size < 2) return null

        val before = parts[0].trim()
        val after = parts[1].trim()

        if (before.isEmpty() || after.isEmpty()) return null

        // Check if "after" is a zone -> this is FAST, not targeting
        // "log the bridge" -> card=log, zone=bridge
        val (afterZone, _) = extractZone(after)
        if (afterZone != null) {
            val cardMatch = matchCard(stripFillers(before), deckCards)
            if (cardMatch != null) {
                return ParsedCommand(
                    tier = CommandTier.FAST,
                    card = cardMatch.first,
                    zone = afterZone,
                    confidence = cardMatch.second,
                    rawTranscript = rawTranscript
                )
            }
        }

        // Check if "after" matches a card name -> this is TARGETING
        // "fireball the hog rider" -> card=fireball, target=hog rider
        // Match against ALL cards (target can be any card, not just deck)
        val allCards = CardAliases.CARD_ALIASES.values.toSet().toList()
        val targetMatch = matchCard(after, allCards)
        if (targetMatch != null) {
            val cardMatch = matchCard(stripFillers(before), deckCards)
            if (cardMatch != null) {
                return ParsedCommand(
                    tier = CommandTier.TARGETING,
                    card = cardMatch.first,
                    targetCard = targetMatch.first,
                    confidence = cardMatch.second,
                    rawTranscript = rawTranscript
                )
            }
        }

        // Neither zone nor card after "the" — fall through
        return null
    }

    // ── Normal flow (steps 7-15 from plan) ──────────────────────────────

    private fun parseNormalCommand(
        segment: String,
        deckCards: List<String>,
        rawTranscript: String
    ): ParsedCommand {
        // Step 7: Extract zone
        val (zone, remaining) = extractZone(segment)

        // Step 8: Strip fillers (FIX-3)
        val cleaned = stripFillers(remaining)

        // Step 9: Match card (FIX-6: multi-level)
        val cardMatch = matchCard(cleaned, deckCards)

        // Step 10: No card matched -> SMART path
        if (cardMatch == null) {
            return ParsedCommand(
                tier = CommandTier.SMART,
                rawTranscript = rawTranscript
            )
        }

        val (card, confidence) = cardMatch

        // Step 11: Card + zone -> FAST
        if (zone != null) {
            return ParsedCommand(
                tier = CommandTier.FAST,
                card = card,
                zone = zone,
                confidence = confidence,
                rawTranscript = rawTranscript
            )
        }

        // Step 12: Card only, check default zone (FIX-2)
        val defaultZone = getDefaultZone(card)

        // Step 13: 5+ elixir with no zone -> reject (route to SMART with hint)
        if (defaultZone == null) {
            return ParsedCommand(
                tier = CommandTier.SMART,
                card = card,
                confidence = confidence,
                rawTranscript = "Where? Say '${card} left/right/center'"
            )
        }

        // Step 14: Card with default zone -> FAST
        return ParsedCommand(
            tier = CommandTier.FAST,
            card = card,
            zone = defaultZone,
            confidence = confidence,
            rawTranscript = rawTranscript
        )
    }

    // ── Conditional parsing (stub for M9) ───────────────────────────────

    private fun parseConditional(
        body: String,
        deckCards: List<String>,
        rawTranscript: String
    ): ParsedCommand {
        // Pattern: "if [trigger_card] [drop/play] [response_card] [zone]"
        // e.g. "if hog drop cannon center"
        // For M3: parse what we can, stub execution

        // Try to find "drop"/"play" as separator
        val separators = listOf(" drop ", " play ", " use ", " then ")
        for (sep in separators) {
            if (body.contains(sep)) {
                val parts = body.split(sep, limit = 2)
                val triggerText = parts[0].trim()
                val responseText = parts[1].trim()

                val allCards = CardAliases.CARD_ALIASES.values.toSet().toList()
                val triggerMatch = matchCard(triggerText, allCards)
                val (zone, responseRemaining) = extractZone(responseText)
                val responseMatch = matchCard(stripFillers(responseRemaining), deckCards)

                if (triggerMatch != null && responseMatch != null) {
                    return ParsedCommand(
                        tier = CommandTier.CONDITIONAL,
                        card = responseMatch.first,
                        zone = zone ?: getDefaultZone(responseMatch.first),
                        triggerCard = triggerMatch.first,
                        confidence = responseMatch.second,
                        rawTranscript = rawTranscript
                    )
                }
            }
        }

        // Couldn't parse conditional fully — route to SMART
        return ParsedCommand(
            tier = CommandTier.CONDITIONAL,
            rawTranscript = rawTranscript
        )
    }

    // ── Queue parsing (stub for M7) ─────────────────────────────────────

    private fun parseQueueCommand(
        body: String,
        deckCards: List<String>,
        rawTranscript: String
    ): ParsedCommand {
        // Pattern: "[card] [zone]"
        val (zone, remaining) = extractZone(body)
        val cleaned = stripFillers(remaining)
        val cardMatch = matchCard(cleaned, deckCards)

        if (cardMatch != null) {
            val defaultZone = zone ?: getDefaultZone(cardMatch.first)
            return ParsedCommand(
                tier = CommandTier.QUEUE,
                card = cardMatch.first,
                zone = defaultZone,
                confidence = cardMatch.second,
                rawTranscript = rawTranscript
            )
        }

        return ParsedCommand(
            tier = CommandTier.QUEUE,
            rawTranscript = rawTranscript
        )
    }
}
