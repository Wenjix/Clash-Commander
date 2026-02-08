package com.yoyostudios.clashcompanion.util

/**
 * Card alias dictionary, elixir costs, card types, compound misrecognitions,
 * and filler words. Pure data — no Android dependencies.
 *
 * Sources:
 *  - Official card names from cr-api-data
 *  - Abbreviations from Clash Royale community slang
 *  - STT misrecognitions from 600+ utterance testing on Samsung A35
 *    with Zipformer transducer + hotword biasing
 */
object CardAliases {

    // ── Filler words to strip before card matching ──────────────────────
    val FILLER_WORDS: Set<String> = setOf(
        "play", "drop", "put", "use", "place", "throw",
        "the", "a", "my", "do", "go", "at", "on", "in", "to",
        "with", "for", "and", "or", "just", "now", "please",
        "can", "you", "it", "that", "this"
    )

    // ── Non-command phrases (exact match after cleaning) ────────────────
    val NON_COMMANDS: Set<String> = setOf(
        "the", "uh", "um", "a", "an", "and", "or", "is", "it", "i",
        "oh", "hmm", "ah", "okay", "ok", "yeah", "yes", "no",
        "hey", "hi", "hello", "what", "huh", "so", "like", "well"
    )

    // ── Compound misrecognitions: STT merges card+zone into one word ────
    // Checked BEFORE zone extraction. Maps to (card, zone_key).
    val COMPOUND_ALIASES: Map<String, Pair<String, String>> = mapOf(
        // "knight right" -> heard as "nitrate" 100% of the time
        "nitrate" to ("Knight" to "right_bridge"),
        "nitrite" to ("Knight" to "right_bridge"),
        "night rate" to ("Knight" to "right_bridge"),
        "night right" to ("Knight" to "right_bridge"),
        "nightright" to ("Knight" to "right_bridge"),
        // "knight left"
        "night left" to ("Knight" to "left_bridge"),
        "nightleft" to ("Knight" to "left_bridge"),
        // "knight center"
        "night center" to ("Knight" to "center"),
        "night middle" to ("Knight" to "center"),
        // "loon bridge" -> heard as "loombridge"
        "loombridge" to ("Balloon" to "left_bridge"),
        "loom bridge" to ("Balloon" to "left_bridge"),
        // "arrows left" -> sometimes heard as "i was left"
        "i was left" to ("Arrows" to "left_bridge"),
        "i was right" to ("Arrows" to "right_bridge"),
        // "mini pekka right" -> STT merges "pekka right" into "pek aright"
        "mini pek aright" to ("Mini P.E.K.K.A" to "right_bridge"),
        "pek aright" to ("Mini P.E.K.K.A" to "right_bridge"),
        "mini pek aleft" to ("Mini P.E.K.K.A" to "left_bridge"),
        "pek aleft" to ("Mini P.E.K.K.A" to "left_bridge"),
        // "pekka right" -> STT produces "pecarite"
        "pecarite" to ("Mini P.E.K.K.A" to "right_bridge"),
        "pecaright" to ("Mini P.E.K.K.A" to "right_bridge"),
        "pecaleft" to ("Mini P.E.K.K.A" to "left_bridge"),
        "pek a lift" to ("Mini P.E.K.K.A" to "left_bridge"),
        // STT merges everything into one word
        "peklift" to ("Mini P.E.K.K.A" to "left_bridge"),
        "pecalift" to ("Mini P.E.K.K.A" to "left_bridge"),
        "pekleft" to ("Mini P.E.K.K.A" to "left_bridge"),
        "pekright" to ("Mini P.E.K.K.A" to "right_bridge"),
        "pecarite" to ("Mini P.E.K.K.A" to "right_bridge"),
        // "minions left" -> STT truncates to "minift"
        "minift" to ("Minions" to "left_bridge"),
        // "musketeer right" -> STT merges into one word
        "musketeeroit" to ("Musketeer" to "right_bridge"),
        "musketeerite" to ("Musketeer" to "right_bridge"),
        "musketeerleft" to ("Musketeer" to "left_bridge"),
        // "mini pekka right" -> STT produces "mini peckerite"
        "mini peckerite" to ("Mini P.E.K.K.A" to "right_bridge"),
        "mini peckerite" to ("Mini P.E.K.K.A" to "right_bridge"),
        "mini peckered" to ("Mini P.E.K.K.A" to "right_bridge"),
    )

    // ── Card aliases: lowercase alias -> proper-case official name ──────
    // Includes official names, community slang, and Zipformer misrecognitions.
    val CARD_ALIASES: Map<String, String> = mapOf(
        // ══════════════════════════════════════════════════
        // DEMO DECK (Knight, Archers, Minions, Arrows,
        //            Fireball, Giant, Musketeer, P.E.K.K.A)
        // ══════════════════════════════════════════════════
        "knight" to "Knight",
        "night" to "Knight",           // STT: 100% misrecognition
        "nite" to "Knight",
        "nights" to "Knight",

        "archers" to "Archers",
        "archer" to "Archers",

        "minions" to "Minions",
        "minion" to "Minions",

        "arrows" to "Arrows",
        "arrow" to "Arrows",
        "ours" to "Arrows",            // STT: "arrows" -> "ours"
        "hours" to "Arrows",           // STT variant

        "fireball" to "Fireball",
        "fire ball" to "Fireball",
        "fire" to "Fireball",

        "giant" to "Giant",
        "giants" to "Giant",           // STT: "giant's" cleaned to "giants"

        "musketeer" to "Musketeer",
        "musky" to "Musketeer",
        "muskee" to "Musketeer",       // STT misrecognition
        "muskey" to "Musketeer",       // STT misrecognition
        "musket" to "Musketeer",
        "musk" to "Musketeer",

        "pekka" to "P.E.K.K.A",
        "p.e.k.k.a" to "P.E.K.K.A",
        "pecker" to "P.E.K.K.A",      // STT misrecognition
        "peker" to "P.E.K.K.A",       // STT misrecognition
        "peca" to "P.E.K.K.A",
        "pega" to "P.E.K.K.A",        // STT misrecognition (Moonshine era)
        "pek" to "P.E.K.K.A",         // STT: "mini pek aright"
        "pecar" to "P.E.K.K.A",       // STT: "pecarite" fragment
        "pack" to "P.E.K.K.A",        // STT: "pekka" -> "pack"
        "packer" to "P.E.K.K.A",      // STT: "pekka" -> "packer"
        "peck" to "P.E.K.K.A",        // STT: "pekka" -> "peck"

        // ══════════════════════════════════════════════════
        // ALL OTHER CARDS (alphabetical)
        // ══════════════════════════════════════════════════

        // -- A --
        "archer queen" to "Archer Queen",
        "baby dragon" to "Baby Dragon",
        "baby d" to "Baby Dragon",
        "bandit" to "Bandit",
        "barbarians" to "Barbarians",
        "barbs" to "Barbarians",
        "barbarian barrel" to "Barbarian Barrel",
        "barb barrel" to "Barbarian Barrel",
        "barbarian hut" to "Barbarian Hut",
        "barb hut" to "Barbarian Hut",
        "bats" to "Bats",
        "bat" to "Bats",
        "battle healer" to "Battle Healer",
        "healer" to "Battle Healer",
        "battle ram" to "Battle Ram",
        "ram" to "Battle Ram",
        "bomber" to "Bomber",
        "bomb tower" to "Bomb Tower",
        "bowler" to "Bowler",

        // -- C --
        "cannon" to "Cannon",
        "cannon cart" to "Cannon Cart",
        "cannoneer" to "Cannoneer",
        "clone" to "Clone",

        // -- D --
        "dagger duchess" to "Dagger Duchess",
        "duchess" to "Dagger Duchess",
        "dark prince" to "Dark Prince",
        "dp" to "Dark Prince",
        "dart goblin" to "Dart Goblin",

        // -- E --
        "earthquake" to "Earthquake",
        "eq" to "Earthquake",
        "electro dragon" to "Electro Dragon",
        "e drag" to "Electro Dragon",
        "edrag" to "Electro Dragon",
        "electro giant" to "Electro Giant",
        "e giant" to "Electro Giant",
        "egiant" to "Electro Giant",
        "electro spirit" to "Electro Spirit",
        "e spirit" to "Electro Spirit",
        "electro wizard" to "Electro Wizard",
        "e wiz" to "Electro Wizard",
        "ewiz" to "Electro Wizard",
        "he was" to "Electro Wizard",  // STT: "e wiz back" -> "he was back"
        "elite barbarians" to "Elite Barbarians",
        "e barbs" to "Elite Barbarians",
        "ebarbs" to "Elite Barbarians",
        "elixir collector" to "Elixir Collector",
        "collector" to "Elixir Collector",
        "pump" to "Elixir Collector",
        "elixir golem" to "Elixir Golem",
        "executioner" to "Executioner",
        "exe" to "Executioner",

        // -- F --
        "fire spirit" to "Fire Spirit",
        "firecracker" to "Firecracker",
        "fisherman" to "Fisherman",
        "flying machine" to "Flying Machine",
        "freeze" to "Freeze",
        "furnace" to "Furnace",

        // -- G --
        "giant skeleton" to "Giant Skeleton",
        "giant skelly" to "Giant Skeleton",
        "goblin barrel" to "Goblin Barrel",
        "gob barrel" to "Goblin Barrel",
        "golden barrel" to "Goblin Barrel", // STT misrecognition
        "goblin cage" to "Goblin Cage",
        "goblin drill" to "Goblin Drill",
        "drill" to "Goblin Drill",
        "goblin gang" to "Goblin Gang",
        "gang" to "Goblin Gang",
        "goblin hut" to "Goblin Hut",
        "goblin machine" to "Goblin Machine",
        "goblins" to "Goblins",
        "goblin" to "Goblins",
        "golden knight" to "Golden Knight",
        "golem" to "Golem",
        "graveyard" to "Graveyard",
        "gy" to "Graveyard",
        "guards" to "Guards",
        "guard" to "Guards",

        // -- H --
        "heal spirit" to "Heal Spirit",
        "hog rider" to "Hog Rider",
        "hog" to "Hog Rider",
        "hug" to "Hog Rider",          // STT misrecognition
        "hunter" to "Hunter",

        // -- I --
        "ice golem" to "Ice Golem",
        "ice spirit" to "Ice Spirit",
        "ice wizard" to "Ice Wizard",
        "inferno dragon" to "Inferno Dragon",
        "inferno drag" to "Inferno Dragon",
        "i drag" to "Inferno Dragon",
        "inferno tower" to "Inferno Tower",
        "inferno" to "Inferno Tower",

        // -- K --
        "knight" to "Knight", // already above, duplicate harmless in mapOf

        // -- L --
        "lava hound" to "Lava Hound",
        "lava" to "Lava Hound",
        "love ahound" to "Lava Hound",   // STT misrecognition
        "lovehound" to "Lava Hound",     // STT misrecognition
        "lightning" to "Lightning",
        "little prince" to "Little Prince",
        "log" to "The Log",
        "the log" to "The Log",
        "lumberjack" to "Lumberjack",
        "lj" to "Lumberjack",

        // -- M --
        "magic archer" to "Magic Archer",
        "ma" to "Magic Archer",
        "mega knight" to "Mega Knight",
        "mk" to "Mega Knight",
        "megaite" to "Mega Knight",     // STT misrecognition
        "megainite" to "Mega Knight",   // STT misrecognition
        "mega minion" to "Mega Minion",
        "mighty miner" to "Mighty Miner",
        "miner" to "Miner",
        "mini pekka" to "Mini P.E.K.K.A",
        "mini pekka" to "Mini P.E.K.K.A",
        "mini p" to "Mini P.E.K.K.A",
        "mirror" to "Mirror",
        "monk" to "Monk",
        "mortar" to "Mortar",
        "mother witch" to "Mother Witch",

        // -- N --
        "night witch" to "Night Witch",

        // -- P --
        "phoenix" to "Phoenix",
        "poison" to "Poison",
        "prince" to "Prince",

        // -- R --
        "rage" to "Rage",
        "rage spirit" to "Rage Spirit",
        "ram rider" to "Ram Rider",
        "rascals" to "Rascals",
        "rocket" to "Rocket",
        "royal delivery" to "Royal Delivery",
        "royal ghost" to "Royal Ghost",
        "ghost" to "Royal Ghost",
        "royal giant" to "Royal Giant",
        "rg" to "Royal Giant",
        "royal hogs" to "Royal Hogs",
        "royal recruits" to "Royal Recruits",
        "recruits" to "Royal Recruits",

        // -- S --
        "skeleton army" to "Skeleton Army",
        "skarmy" to "Skeleton Army",
        "scar me" to "Skeleton Army",   // STT: "skarmy right" -> "scar me right"
        "scary" to "Skeleton Army",     // STT misrecognition
        "skull army" to "Skeleton Army", // STT misrecognition
        "skeleton barrel" to "Skeleton Barrel",
        "skeleton dragons" to "Skeleton Dragons",
        "skeleton king" to "Skeleton King",
        "skelly king" to "Skeleton King",
        "skeletons" to "Skeletons",
        "snowball" to "Snowball",
        "sparky" to "Sparky",
        "spear goblins" to "Spear Goblins",
        "spear gobs" to "Spear Goblins",

        // -- T --
        "tesla" to "Tesla",
        "three musketeers" to "Three Musketeers",
        "3m" to "Three Musketeers",
        "tombstone" to "Tombstone",
        "tornado" to "Tornado",
        "nado" to "Tornado",
        "tower princess" to "Tower Princess",

        // -- V --
        "valkyrie" to "Valkyrie",
        "valk" to "Valkyrie",

        // -- W --
        "wall breakers" to "Wall Breakers",
        "wb" to "Wall Breakers",
        "witch" to "Witch",
        "wizard" to "Wizard",
        "wiz" to "Wizard",

        // -- X --
        "x-bow" to "X-Bow",
        "x bow" to "X-Bow",
        "xbow" to "X-Bow",
        "expo" to "X-Bow",            // STT: "x bow" -> "expo" 62%
        "crossbow" to "X-Bow",

        // -- Z --
        "zap" to "Zap",
        "zappies" to "Zappies",

        // ── Additional STT misrecognitions ──
        "balloon" to "Balloon",
        "loon" to "Balloon",
        "blue" to "Balloon",           // STT misrecognition
        "loom" to "Balloon",           // STT: "loon bridge" remnant
    )

    // ── Elixir costs for ALL cards ──────────────────────────────────────
    val CARD_ELIXIR: Map<String, Int> = mapOf(
        // 1 elixir
        "Skeletons" to 1,
        "Ice Spirit" to 1,
        "Fire Spirit" to 1,
        "Heal Spirit" to 1,
        "Electro Spirit" to 1,
        "Rage Spirit" to 1,

        // 2 elixir
        "Goblins" to 2,
        "Spear Goblins" to 2,
        "Bats" to 2,
        "Zap" to 2,
        "Snowball" to 2,
        "The Log" to 2,
        "Barbarian Barrel" to 2,
        "Wall Breakers" to 2,
        "Bomber" to 2,
        "Rage" to 2,
        "Tombstone" to 2,

        // 3 elixir
        "Knight" to 3,
        "Archers" to 3,
        "Minions" to 3,
        "Arrows" to 3,
        "Cannon" to 3,
        "Goblin Gang" to 3,
        "Goblin Barrel" to 3,
        "Dart Goblin" to 3,
        "Guards" to 3,
        "Miner" to 3,
        "Bandit" to 3,
        "Fisherman" to 3,
        "Royal Ghost" to 3,
        "Tornado" to 3,
        "Clone" to 3,
        "Mirror" to 3,
        "Ice Golem" to 3,
        "Mega Minion" to 3,
        "Skeleton Barrel" to 3,
        "Goblin Cage" to 3,
        "Firecracker" to 3,
        "Earthquake" to 3,
        "Tesla" to 3,
        "Mortar" to 3,
        "Monk" to 3,

        // 4 elixir
        "Musketeer" to 4,
        "Fireball" to 4,
        "Mini P.E.K.K.A" to 4,
        "Valkyrie" to 4,
        "Hog Rider" to 4,
        "Battle Ram" to 4,
        "Dark Prince" to 4,
        "Prince" to 4,
        "Baby Dragon" to 4,
        "Hunter" to 4,
        "Poison" to 4,
        "Freeze" to 4,
        "Furnace" to 4,
        "Goblin Hut" to 4,
        "Goblin Drill" to 4,
        "Bomb Tower" to 4,
        "Flying Machine" to 4,
        "Zappies" to 4,
        "Cannon Cart" to 4,
        "Skeleton Dragons" to 4,
        "Inferno Dragon" to 4,
        "Mother Witch" to 4,
        "Night Witch" to 4,
        "Lumberjack" to 4,
        "Magic Archer" to 4,
        "Ram Rider" to 4,
        "Royal Delivery" to 4,
        "Barbarians" to 4,
        "Electro Wizard" to 4,
        "Golden Knight" to 4,
        "Skeleton King" to 4,
        "Phoenix" to 4,
        "Little Prince" to 4,
        "Tower Princess" to 4,
        "Dagger Duchess" to 4,
        "Cannoneer" to 4,
        "Battle Healer" to 4,
        "Goblin Machine" to 4,

        // 5 elixir
        "Giant" to 5,
        "Balloon" to 5,
        "Witch" to 5,
        "Wizard" to 5,
        "Bowler" to 5,
        "Executioner" to 5,
        "Inferno Tower" to 5,
        "Elixir Collector" to 5,
        "Rascals" to 5,
        "Royal Hogs" to 5,
        "Mighty Miner" to 5,
        "Graveyard" to 5,
        "Sparky" to 5,
        "Archer Queen" to 5,
        "X-Bow" to 5,

        // 6 elixir
        "Royal Giant" to 6,
        "Giant Skeleton" to 6,
        "Lightning" to 6,
        "Rocket" to 6,
        "Elite Barbarians" to 6,
        "Barbarian Hut" to 6,
        "Electro Giant" to 6,
        "Skeleton Army" to 6,
        "Royal Recruits" to 6,
        "Elixir Golem" to 6,
        "Electro Dragon" to 6,

        // 7 elixir
        "P.E.K.K.A" to 7,
        "Mega Knight" to 7,
        "Lava Hound" to 7,
        "Royal Giant" to 6,

        // 9 elixir
        "Three Musketeers" to 9,
        "Golem" to 8,
    )

    // ── Card types ──────────────────────────────────────────────────────
    val CARD_TYPES: Map<String, String> = mapOf(
        // Troops
        "Knight" to "troop",
        "Archers" to "troop",
        "Minions" to "troop",
        "Giant" to "troop",
        "Musketeer" to "troop",
        "P.E.K.K.A" to "troop",
        "Hog Rider" to "troop",
        "Balloon" to "troop",
        "Skeleton Army" to "troop",
        "Witch" to "troop",
        "Wizard" to "troop",
        "Bomber" to "troop",
        "Barbarians" to "troop",
        "Golem" to "troop",
        "Sparky" to "troop",
        "Bandit" to "troop",
        "Lumberjack" to "troop",
        "Valkyrie" to "troop",
        "Prince" to "troop",
        "Dark Prince" to "troop",
        "Baby Dragon" to "troop",
        "Electro Wizard" to "troop",
        "Electro Dragon" to "troop",
        "Electro Giant" to "troop",
        "Inferno Dragon" to "troop",
        "Night Witch" to "troop",
        "Lava Hound" to "troop",
        "Mega Knight" to "troop",
        "Elite Barbarians" to "troop",
        "Royal Giant" to "troop",
        "Royal Hogs" to "troop",
        "Royal Ghost" to "troop",
        "Three Musketeers" to "troop",
        "Mini P.E.K.K.A" to "troop",
        "Flying Machine" to "troop",
        "Magic Archer" to "troop",
        "Wall Breakers" to "troop",
        "Ram Rider" to "troop",
        "Battle Ram" to "troop",
        "Ice Wizard" to "troop",
        "Ice Golem" to "troop",
        "Mega Minion" to "troop",
        "Cannon Cart" to "troop",
        "Mother Witch" to "troop",
        "Golden Knight" to "troop",
        "Mighty Miner" to "troop",
        "Archer Queen" to "troop",
        "Skeleton King" to "troop",
        "Little Prince" to "troop",
        "Tower Princess" to "troop",
        "Dagger Duchess" to "troop",
        "Skeleton Barrel" to "troop",
        "Skeleton Dragons" to "troop",
        "Giant Skeleton" to "troop",
        "Bowler" to "troop",
        "Executioner" to "troop",
        "Hunter" to "troop",
        "Fisherman" to "troop",
        "Guards" to "troop",
        "Bats" to "troop",
        "Goblins" to "troop",
        "Spear Goblins" to "troop",
        "Goblin Gang" to "troop",
        "Dart Goblin" to "troop",
        "Skeletons" to "troop",
        "Ice Spirit" to "troop",
        "Fire Spirit" to "troop",
        "Heal Spirit" to "troop",
        "Electro Spirit" to "troop",
        "Rage Spirit" to "troop",
        "Miner" to "troop",
        "Rascals" to "troop",
        "Firecracker" to "troop",
        "Phoenix" to "troop",
        "Monk" to "troop",
        "Cannoneer" to "troop",
        "Zappies" to "troop",
        "Battle Healer" to "troop",
        "Elixir Golem" to "troop",
        "Royal Recruits" to "troop",
        "Royal Delivery" to "troop",
        "Goblin Machine" to "troop",

        // Spells
        "Arrows" to "spell",
        "Fireball" to "spell",
        "Zap" to "spell",
        "The Log" to "spell",
        "Snowball" to "spell",
        "Barbarian Barrel" to "spell",
        "Lightning" to "spell",
        "Rocket" to "spell",
        "Tornado" to "spell",
        "Earthquake" to "spell",
        "Poison" to "spell",
        "Freeze" to "spell",
        "Rage" to "spell",
        "Clone" to "spell",
        "Mirror" to "spell",
        "Graveyard" to "spell",

        // Buildings
        "Cannon" to "building",
        "Tesla" to "building",
        "Mortar" to "building",
        "Inferno Tower" to "building",
        "Bomb Tower" to "building",
        "Tombstone" to "building",
        "Furnace" to "building",
        "Goblin Hut" to "building",
        "Goblin Cage" to "building",
        "Goblin Drill" to "building",
        "Barbarian Hut" to "building",
        "Elixir Collector" to "building",
        "X-Bow" to "building",
    )
}
