import fandom

fandom.set_wiki('Witcher')
fandom.set_lang('en')
geralt = fandom.search("Avallach")
geralt = geralt[0][0]
geralt = fandom.page(geralt)
print(geralt.title)
print(geralt.url)
print(geralt.plain_text)


# witcher_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Yennefer of Vengerberg', 'Ciri (Cirilla Fiona Elen Riannon)', 
#     'Triss Merigold', 'Vesemir', 'Dandelion (Jaskier)', 'Zoltan Chivay', 
#     'Emhyr var Emreis', 'Regis', 'Eskel', 'Lambert', 'Leo', 'Milva', 
#     'Angoulême', 'Cahir Mawr Dyffryn aep Ceallach', 'Bonhart', 'Mousesack',

#     # Witcher Schools
#     'School of the Wolf', 'School of the Cat', 'School of the Griffin', 
#     'School of the Bear', 'School of the Viper', 'School of the Manticore', 'School of the Crane',

#     # Antagonists
#     'Eredin Bréacc Glas', 'Imlerith', 'Caranthir', 'Gaunter O\'Dimm', 
#     'Dettlaff van der Eretein', 'Rience', 'Vilgefortz', 'Letho of Gulet', 
#     'The Crones of Crookback Bog (Brewess, Weavess, Whispess)', 
#     'Leo Bonhart', 'Stefan Skellen', 'Renfri of Creyden', 'Falwick of Malleore',

#     # Creatures and Monsters
#     'Leshen', 'Griffin', 'Wyvern', 'Drowner', 'Golem', 'Chort', 'Striga', 
#     'Kikimora', 'Alghoul', 'Ekimmara', 'Bruxa', 'Werewolf', 'Djinn', 
#     'Foglet', 'Hym', 'Noonwraith', 'Nightwraith', 'Basilisk', 'Succubus',
#     'Vampires', 'Fiend', 'Cyclops', 'Water Hag', 'Grave Hag', 'Cave Troll',
#     'Ice Giant', 'Ekimma', 'Godling', 'Higher Vampire', 'Sylvan',

#     # Locations
#     'Kaer Morhen', 'Novigrad', 'Vizima', 'Oxenfurt', 'Toussaint', 
#     'Skellige Isles', 'Velen', 'Beauclair', 'Brokilon Forest', 'Cintra', 
#     'Aretuza', 'Ban Ard', 'Rivia', 'Temeria', 'Redania', 'Nilfgaard',
#     'Pontar Valley', 'Tretogor', 'Gors Velen', 'Ellander', 'Dol Blathanna',
#     'Brenna', 'Ebbing', 'Metinna', 'Nazair', 'Vicovaro', 'Mahakam Mountains',

#     # Factions and Organizations
#     'Nilfgaardian Empire', 'Scoia\'tael', 'The Lodge of Sorceresses', 
#     'Order of the Flaming Rose', 'Wild Hunt', 'Witch Hunters',
#     'Brotherhood of Sorcerers', 'Crimson Reavers', 'Temerian Secret Service',
#     'The Blue Stripes', 'The Rats', 'Kaedwen', 'Aedirn',

#     # Important Characters from Books and Games
#     'Philippa Eilhart', 'Francesca Findabair', 'Keira Metz', 'Vernon Roche', 
#     'Thaler', 'Sigismund Dijkstra', 'Shani', 'Anna Henrietta', 
#     'Queen Calanthe', 'Eithné', 'Avallac\'h', 'Coral (Lytta Neyd)', 'Sile de Tansarville',
#     'Isengrim Faoiltiarna', 'Toruviel', 'Detmold', 'Foltest', 'Radovid V',
#     'Adda the White', 'Henselt', 'Vattier de Rideaux',

#     # Lore and Items
#     'Elder Blood', 'Zireael Sword', 'The Law of Surprise', 
#     'Gwent', 'White Frost', 'Trial of the Grasses', 'Lilac and Gooseberries',
#     'Aerondight', 'Dimeritium Shackles', 'Moon Dust Bomb', 'Silver Sword', 
#     'Wolven Armor', 'Mutagens', 'Potion of Thunderbolt', 'Swallow Potion',

#     # Spells and Abilities
#     'Aard', 'Igni', 'Yrden', 'Quen', 'Axii',
#     'Portals', 'Alchemy', 'Necromancy', 'Illusion', 'Fireball',

#     # Books by Andrzej Sapkowski
#     'The Last Wish', 'Sword of Destiny', 'Blood of Elves', 
#     'Time of Contempt', 'Baptism of Fire', 'The Tower of the Swallow', 
#     'The Lady of the Lake', 'Season of Storms',

#     # Trivia and Easter Eggs
#     'Butcher of Blaviken', 'White Wolf', 'Lilies and Roses of the Valley', 
#     'Elder Speech', 'Flotsam', 'The Lady of the Lake (myth)', 
#     'Hen Gaidth', 'Dol Durza',

#     # Adaptations
#     'The Witcher Netflix series', 'The Witcher 3: Wild Hunt', 'Hearts of Stone', 
#     'Blood and Wine', 'The Witcher 2: Assassins of Kings', 'The Witcher (2007 game)', 
#     'The Hexer (Polish series)', 'Witcher graphic novels',

#     # Musical Elements
#     'Priscilla\'s Song (Wolven Storm)', 'The Fields of Ard Skellig', 'Kaer Morhen theme',
#     'Silver for Monsters',

#     # Miscellaneous
#     'White Orchard', 'Eternal Fire', 'Baptism of Fire', 'Shard of Ice', 'The Iron Judgment'
# ]

# witcher_1_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Triss Merigold', 'Vesemir', 'Leo', 'Lambert', 
#     'Eskel', 'Alvin', 'Shani', 'Zoltan Chivay', 'Dandelion (Jaskier)', 
#     'King Foltest', 'Adda the White',

#     # Major Antagonists
#     'Jacques de Aldersberg', 'Azar Javed', 'Professor (leader of Salamandra)', 
#     'Roderick de Wett', 'Magister of the Order', 'Berengar',

#     # Key Factions
#     'Salamandra', 'Order of the Flaming Rose', 'Scoia\'tael', 
#     'Temerian Special Forces',

#     # Notable Locations
#     'Kaer Morhen', 'Vizima', 'Old Vizima', 'Swamp Forest', 
#     'Temple Quarter', 'Trade Quarter', 'Dike', 'Cemetery', 
#     'Vizima Castle', 'Outskirts of Vizima', 'Murky Waters', 
#     'The Fields',

#     # Important Side Characters
#     'Declan Leuvaarden', 'Vincent Meis', 'Vivaldi', 
#     'Raymond Maarloeve', 'Carmen', 'Thaler', 'Patrick de Weyze',
#     'Severin', 'Haren Brogg', 'Julian', 'Dagon Cultists',

#     # Monsters and Creatures
#     'Koshchey', 'Striga (Adda)', 'Ifrit', 'Alghoul', 'Echinops', 
#     'Wyvern', 'Bloedzuiger', 'Graveir', 'Devourer', 
#     'Fleder', 'Drowners', 'Kikimore Queen', 'Beast (Barghest leader)',
#     'Cockatrice', 'Royal Wyvern', 'Golem', 'Cemetery Horror',

#     # Spells and Abilities
#     'Aard', 'Igni', 'Yrden', 'Quen', 'Axii',

#     # Significant Items
#     'Aerondight', 'Raven\'s Armor', 'Moonblade', 'G'valchir', 
#     'Temerian Steel Sword', 'Dagon\'s Secret Weapon', 
#     'Diamond Dust', 'Orens (currency)', 'Mandrake Root',

#     # Alchemy Ingredients and Potions
#     'Swallow Potion', 'White Raffard\'s Decoction', 'Petri\'s Philter', 
#     'Blizzard', 'Cat Potion', 'Kikimore Ichor', 'Vitriol', 'Quebrith', 
#     'Rebis', 'Hydragenum', 'Vermilion', 'Aether',

#     # Lore and Quest Elements
#     'Trial of the Grasses', 'Conjunction of the Spheres', 
#     'Eternal Fire', 'The Witcher Contracts', 'The Beast of the Outskirts', 
#     'Alvin\'s Prophecy', 'The Great War of the North', 'Viziman Politics',

#     # Chapters and Major Plot Points
#     'Prologue - Kaer Morhen', 'Chapter 1 - Outskirts of Vizima', 
#     'Chapter 2 - Temple Quarter', 'Chapter 3 - Trade Quarter', 
#     'Chapter 4 - Murky Waters', 'Chapter 5 - Old Vizima',
#     'Epilogue - Order or Scoia\'tael Paths',

#     # Side Quests
#     'Blue Eyes (Carmen\'s Quest)', 'Six Feet Under', 
#     'Won\'t Hurt a Bit', 'Safe Haven', 'Dead Hand of the Past', 
#     'Suspect: Vivaldi', 'Suspect: Thaler', 'Force Recon',

#     # Music and Audio
#     'Kaer Morhen Theme', 'Peaceful Moments', 'Night in Vizima', 
#     'The Dike at Night', 'Dandelion\'s Ballad',

#     # Miscellaneous
#     'Witcher Medallion', 'Barghests', 'Viziman Wine', 'Eternal Fire Priests', 
#     'Silver Swords', 'Bestiary', 'Monolith Puzzles', 'Dice Poker', 
#     'Fistfighting Tournament', 'Lady of the Lake'
# ]

# witcher_2_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Triss Merigold', 'Vernon Roche', 'Iorveth', 
#     'King Foltest', 'Letho of Gulet', 'Dandelion (Jaskier)', 
#     'Zoltan Chivay', 'Sile de Tansarville', 'Philippa Eilhart', 
#     'Saskia (the Dragon)', 'Dethmold', 'Henselt', 'Sheala de Tancarville',
#     'Aryan La Valette', 'Detmold', 'Margarita Laux-Antille',

#     # Major Antagonists
#     'Letho of Gulet (Kingslayer)', 'Dethmold', 'Henselt', 
#     'Renuald aep Matsen', 'Nilfgaardian Emissaries',

#     # Notable Locations
#     'Flotsam', 'Vergen', 'Kaedwen', 'Loc Muinne', 
#     'Pontar Valley', 'Aedirn', 'Temeria', 'La Valette Castle', 
#     'Shepherds of Lobinden', 'Dwarven Quarter',

#     # Factions
#     'Temerian Special Forces (Blue Stripes)', 'Scoia\'tael', 
#     'Kaedweni Army', 'Nilfgaardian Empire', 'Lodge of Sorceresses',
#     'Order of the Flaming Rose',

#     # Monsters and Creatures
#     'Arachas', 'Harpy', 'Rotfiend', 'Endrega', 'Nekker', 
#     'Bullvore', 'Gargoyle', 'Draug', 'Kayran', 'Wraiths', 
#     'Operator Golem', 'Succubus', 'Troll',

#     # Spells and Abilities
#     'Aard', 'Igni', 'Yrden', 'Quen', 'Axii',

#     # Significant Items
#     'Zireael Sword', 'Kayran Trap', 'Rose of Remembrance', 
#     'Sword of Kaedwen', 'Blood Sword', 'Oathbreaker Armor', 
#     'Enhanced Blue Stripes Combat Jacket', 'Forgotten Vran Sword',

#     # Alchemy and Ingredients
#     'Swallow Potion', 'White Raffard\'s Decoction', 
#     'Golden Oriole', 'Virga', 'Rook', 'Troll Tongue', 
#     'Endrega Venom', 'Rotfiend Blood',

#     # Lore and Plot Elements
#     'Assassination of Kings', 'Conspiracy in Loc Muinne', 
#     'Pontar Valley Politics', 'Scoia\'tael Rebellion', 
#     'Kaedweni Siege of Vergen', 'Battle of Loc Muinne',
#     'The Lodge Conspiracy', 'Nilfgaardian Incursion',

#     # Chapters and Major Plot Points
#     'Prologue - La Valette Castle', 
#     'Chapter 1 - Flotsam and the Pontar Valley', 
#     'Chapter 2 (Roche\'s Path) - Kaedweni Camp', 
#     'Chapter 2 (Iorveth\'s Path) - Vergen', 
#     'Chapter 3 - Loc Muinne',
#     'Final Showdown with Letho',

#     # Side Quests
#     'The Assassins of Kings', 'In the Claws of Madness', 
#     'Mystic River', 'The Harpy Contract', 'The Nekker Contract', 
#     'Malena', 'Poker Face', 'Hung Over', 'The Succubus Quest',

#     # Music and Audio
#     'Assassins of Kings Theme', 'Dwarven Drinking Song', 
#     'Flotsam Forest Theme', 'Vergen Theme', 'Kaedwen Army March',

#     # Miscellaneous
#     'Witcher Medallion', 'Blue Stripes Tattoo', 'Flotsam Prison',
#     'Dice Poker', 'Fistfighting', 'Arm Wrestling', 'Dragonfire',
#     'Loc Muinne Summit', 'Dwarven Mines', 'Nilfgaardian Spies'
# ]

# witcher_3_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Yennefer of Vengerberg', 'Ciri (Cirilla Fiona Elen Riannon)', 
#     'Triss Merigold', 'Vesemir', 'Lambert', 'Eskel', 
#     'Dandelion (Jaskier)', 'Zoltan Chivay', 'Emhyr var Emreis', 
#     'Roche', 'Thaler', 'Sigismund Dijkstra', 'Keira Metz', 
#     'Philippa Eilhart', 'Francesca Findabair', 'Avallac\'h', 
#     'Gaunter O\'Dimm', 'Dettlaff van der Eretein', 'Anna Henrietta', 
#     'Regis', 'Ves', 'Uma (Avallac\'h\'s Curse)', 'Iris von Everec',

#     # Antagonists
#     'Eredin Bréacc Glas', 'Imlerith', 'Caranthir', 'Gaunter O\'Dimm',
#     'Dettlaff van der Eretein', 'Whispess', 'Brewess', 'Weavess',
#     'Horst Borsodi', 'Olgierd von Everec', 'Whoreson Junior',
#     'Radovid V', 'King of Beggars', 'Dudu Biberveldt',

#     # Major Locations
#     'White Orchard', 'Novigrad', 'Velen', 'Oxenfurt', 
#     'Kaer Morhen', 'Skellige Isles', 'Toussaint', 'Beauclair',
#     'Crookback Bog', 'Temple Isle', 'Vizima Royal Palace', 
#     'The Bloody Baron's Estate', 'The Nilfgaardian Camp', 'Undvik',
#     'Ard Skellig', 'Kaer Trolde', 'Fayrlund', 'Isle of Mists',

#     # Factions
#     'Wild Hunt', 'Scoia\'tael', 'Nilfgaardian Empire', 
#     'Temerian Guerrillas', 'Order of the Flaming Rose', 
#     'Witch Hunters', 'Redanian Secret Service', 'Skellige Clans',

#     # Monsters and Creatures
#     'Leshen', 'Fiend', 'Griffin', 'Wyvern', 'Chort', 
#     'Striga', 'Drowner', 'Alghoul', 'Ekimmara', 
#     'Foglet', 'Hym', 'Noonwraith', 'Nightwraith', 
#     'Basilisk', 'Succubus', 'Botchling', 'Godling', 
#     'Higher Vampire', 'Fiend', 'Ice Giant', 'Werewolf', 
#     'Water Hag', 'Grave Hag', 'Katakan', 'Archgriffin',
#     'Shaelmaar', 'Garkain', 'Wraith',

#     # Witcher Contracts
#     'Jenny o\' the Woods', 'The Beast of White Orchard', 
#     'The Griffin from the Highlands', 'Shrieker', 
#     'The White Lady', 'Lord of the Wood', 
#     'Missing Brother', 'The Merry Widow', 'The Creature from Oxenfurt Forest',

#     # Gwent and Minigames
#     'Gwent', 'Nilfgaardian Deck', 'Scoia\'tael Deck', 
#     'Northern Realms Deck', 'Skellige Deck', 'Dice Poker',
#     'Horse Racing', 'Fistfighting', 'Arm Wrestling',

#     # Significant Items
#     'Aerondight', 'Wolf School Gear', 'Griffin School Gear', 
#     'Ursine School Gear', 'Feline School Gear', 'Viper School Gear',
#     'Dimeritium Shackles', 'Moon Dust Bomb', 'Crossbow',
#     'Mastercrafted Silver Sword', 'Potion of Thunderbolt',

#     # Alchemy Ingredients and Potions
#     'Swallow Potion', 'White Raffard\'s Decoction', 
#     'Petri\'s Philter', 'Blizzard', 'Golden Oriole', 
#     'Mandrake Root', 'Arinaria', 'Vitriol', 'Quebrith', 
#     'Rebis', 'Hydragenum', 'Aether',

#     # Expansions
#     'Hearts of Stone', 'Blood and Wine',

#     # Hearts of Stone Plot Elements
#     'Gaunter O\'Dimm (Master Mirror)', 'Olgierd von Everec', 
#     'The Painted World', 'Von Everec Mansion', 'Shani', 
#     'The Caretaker', 'Iris von Everec',

#     # Blood and Wine Plot Elements
#     'Anna Henrietta', 'Dettlaff van der Eretein', 
#     'Regis', 'The Unseen Elder', 'Tesham Mutna', 
#     'Land of a Thousand Fables', 'Shaelmaar', 
#     'Golyat', 'Syanna', 'The Bruxa of Corvo Bianco',

#     # Lore and Plot Elements
#     'Conjunction of the Spheres', 'Elder Blood', 'The Law of Surprise',
#     'Trial of the Grasses', 'White Frost', 'The Curse of the Black Sun',

#     # Spells and Abilities
#     'Aard', 'Igni', 'Yrden', 'Quen', 'Axii',

#     # Music and Audio
#     'The Wolven Storm (Priscilla\'s Song)', 'Kaer Morhen Theme', 
#     'The Fields of Ard Skellig', 'City of Intrigues', 
#     'Toussaint Theme', 'Silver for Monsters',

#     # Side Quests
#     'Family Matters', 'The Tower Outta Nowhere', 
#     'Carnal Sins', 'Reason of State', 'The Last Wish', 
#     'A Towerful of Mice', 'The Cave of Dreams', 'The Lord of Undvik',
#     'There Can Be Only One',

#     # Miscellaneous
#     'Witcher Medallion', 'Nilfgaardian Armor', 'Ciri\'s Unicorn',
#     'Eredin\'s Ship', 'The Lady of the Lake', 'Corvo Bianco Vineyard', 
#     'Dandelion\'s Cabaret', 'Hen Gaidth', 'Wine Wars', 
#     'Mutations', 'Fablesphere', 'Toussaint Tournament',
#     'Geralt and Yennefer\'s Wedding (Land of a Thousand Fables)',
#     'Orianna', 'Unseen Elder', 'Fountain of Oblivion'
# ]

# witcher_3_hearts_of_stone_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Gaunter O\'Dimm (Master Mirror)', 
#     'Olgierd von Everec', 'Iris von Everec', 'Shani', 
#     'Vlodimir von Everec', 'The Caretaker',

#     # Antagonists
#     'Gaunter O\'Dimm (Master Mirror)', 'The Caretaker', 
#     'Wraiths of the Painted World',

#     # Notable Locations
#     'Von Everec Estate', 'Oxenfurt', 'Iris von Everec\'s Painted World', 
#     'The Temple of Lilvani', 'Borsodi Auction House', 
#     'The Alchemy Inn', 'The Herbalist\'s Hut', 'The Eternal Fire Chapel',

#     # Key Quests
#     'Evil’s Soft First Touches', 'Open Sesame!', 'Dead Man’s Party', 
#     'Scenes From a Marriage', 'Whatsoever a Man Soweth...',
#     'A Midnight Clear', 'Rose on a Red Field', 'To Bait a Forktail...',
    
#     # Important Items
#     'The Rose of Remembrance (Iris\' Rose)', 
#     'Viper School Gear (Hearts of Stone)', 
#     'Iris (Olgierd\'s Saber)', 'Ofieri Armor Set', 
#     'Ofieri Saddlebags', 'Enchanted Runewords', 
#     'Dimaritium Shackles (used by Gaunter)', 
#     'The Wreath of Immortelles',

#     # Unique Monsters and Creatures
#     'The Caretaker', 'Iris\' Wraith', 'Vlodimir von Everec\'s Ghost', 
#     'Ofieri Knights', 'Specters in the Painted World',

#     # Significant Characters and NPCs
#     'Borsodi Brothers (Horst and Ewald)', 
#     'The Hostess at Dead Man\'s Party', 
#     'The Rune Wright (Master of Glyphs)', 
#     'Ofieri Merchant', 'Countess Mignole',

#     # Key Lore Elements
#     'The Pact with Gaunter O\'Dimm', 
#     'Von Everec Family Tragedy', 'Runewright Enchantment Rituals',
#     'Olgierd\'s Immortality Curse', 
#     'The Painted World (Memory Realm)', 'Gaunter\'s True Nature',

#     # Runes and Glyphs
#     'Severance', 'Invigoration', 'Replenishment', 
#     'Preservation', 'Deflection', 'Greater Glyph of Warding',

#     # Music and Audio
#     'Hearts of Stone Theme', 'Scenes from a Marriage Music', 
#     'Dead Man\'s Party Theme', 'Gaunter O\'Dimm’s Whistling Tune',

#     # Side Activities
#     'Borsodi Auction Heist', 'Dead Man’s Party Ghost Festivities', 
#     'Rune Crafting with the Runewright',

#     # Important Plot Choices
#     'Choosing to Save or Defeat Gaunter O\'Dimm', 
#     'Helping Olgierd Redeem Himself', 
#     'Preserving or Destroying the Von Everec Family Legacy',

#     # Easter Eggs and Miscellaneous
#     'The Gaunter O\'Dimm Puzzle Challenge', 
#     'References to Faustian Bargains', 
#     'The Mirror Shattering Trick', 'Painting Memories Puzzle',
#     'Olgierd von Everec\'s Curse Redemption'
# ]

# witcher_3_blood_and_wine_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Anna Henrietta', 'Damien de la Tour',
#     'Regis (Emiel Regis Rohellec Terzieff-Godefroy)', 'Syanna (Sylvia Anna)', 
#     'Dettlaff van der Eretein', 'Orianna', 'Duchess Anna Henrietta',
#     'The Unseen Elder', 'The Bootblack',

#     # Notable Locations
#     'Toussaint', 'Beauclair', 'Corvo Bianco Vineyard', 'Dun Tynne Castle', 
#     'Tesham Mutna', 'The Land of a Thousand Fables', 
#     'Fox Hollow', 'Flovive', 'Bokler Market', 'Palace Gardens',
#     'Chuchote Cave', 'Beauclair Palace', 'Count de la Croix Pass',

#     # Key Quests
#     'Envoys, Wineboys', 'The Beast of Toussaint', 
#     'Blood Run', 'The Night of Long Fangs', 
#     'Beyond Hill and Dale...', 'Tesham Mutna',
#     'Burlap is the New Stripe', 'The Warble of a Smitten Knight', 
#     'Wine Wars: Vermentino & Coronata', 'There Can Be Only One',
#     'La Cage au Fou', 'The Man from Cintra',
    
#     # Important Items and Gear
#     'Aerondight (Sword of Legend)', 
#     'Toussaint Knight\'s Steel Sword', 'Manticore School Gear', 
#     'Grandmaster Witcher Gear', 'Beauclair Gwent Cards', 
#     'Mutagen Extractor', 'Syanna’s Ribbon', 
#     'Dettlaff’s Black Crystal Heart', 'Toussaint Crossbow',

#     # Unique Monsters and Creatures
#     'Scolopendromorph', 'Shaelmaar', 'Garkain', 'Bruxa',
#     'Alp', 'Archespore', 'Wight', 'Golyat', 
#     'The Unseen Elder', 'Higher Vampires', 'Protofiend', 
#     'Land of a Thousand Fables Giants',

#     # Important Characters
#     'Barnabas-Basil Foulty (Vineyard Butler)', 
#     'The Lady of the Lake', 'Hermit of the Lake', 
#     'Marquise Serenity', 'The Five Virtues Guardians', 
#     'Captain de la Tour', 'Artorius Vigo',

#     # Significant Plot Choices
#     'Spare or Kill Dettlaff', 'Save or Betray Syanna', 
#     'Peace or Chaos Ending', 'The Ribbon in the Fablesphere', 
#     'Spending Eternity with Regis', 
#     'Becoming a Knight of Toussaint',

#     # Key Lore Elements
#     'The Curse of the Black Sun', 'The Five Chivalric Virtues of Toussaint',
#     'The Beast of Toussaint Mystery', 
#     'The Legend of Aerondight', 
#     'Higher Vampire Society', 'Tesham Mutna Rituals',
#     'The Tale of the Land of a Thousand Fables',

#     # Spells and Abilities
#     'Mutations (Blood and Wine)', 'Strengthened Quen', 
#     'Piercing Cold Mutation', 'Deadly Counter Mutation', 
#     'Cat Eyes Mutation', 'Euphoria Mutation',

#     # Music and Audio
#     'Blood and Wine Main Theme', 'Beauclair Theme', 
#     'Lady of the Lake Theme', 'Land of a Thousand Fables Music',
#     'Tesham Mutna Ritual Music', 'The Night of Long Fangs Theme',

#     # Side Activities and Minigames
#     'Gwent Tournament at the Pheasantry', 'Wine Wars',
#     'Grandmaster Witcher Gear Hunts', 
#     'Toussaint Knight Tournament', 'Mutations Crafting',

#     # Side Quests
#     'Paperchase', 'Mutual of Beauclair\'s Wild Kingdom', 
#     'Father Knows Worst', 'Goodness, Gracious, Great Balls of Granite!', 
#     'A Knight’s Tales', 'Big Game Hunter', 
#     'Feet as Cold as Ice', 'The Hunger Game',

#     # Easter Eggs and Miscellaneous
#     'The Land of a Thousand Fables (Fairytale World)', 
#     'References to Dark Souls (Shaelmaar)', 
#     'Lady of the Lake Returning Aerondight', 
#     'The Tournament’s Unicorn Joust', 
#     'Beauclair Vineyard Upgrades', 'Magical Spoon Wight Quest',
#     'Slippery Slope of Fablespheres', 'The Illustrious Gwent Cards of Toussaint'
# ]

# witcher_books_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Yennefer of Vengerberg', 'Ciri (Cirilla Fiona Elen Riannon)',
#     'Dandelion (Jaskier)', 'Triss Merigold', 'Vesemir', 'Emhyr var Emreis',
#     'Milva (Maria Barring)', 'Regis (Emiel Regis Rohellec Terzieff-Godefroy)',
#     'Cahir Mawr Dyffryn aep Ceallach', 'Zoltan Chivay', 'Leo Bonhart',
#     'Eithné (Queen of the Dryads)', 'Angoulême', 'Vilgefortz of Roggeveen',
#     'Fringilla Vigo', 'Philippa Eilhart', 'Margarita Laux-Antille',
#     'Tissaia de Vries', 'Sabrina Glevissig', 'Assire var Anahid', 
#     'Coral (Lytta Neyd)', 'Dijkstra', 'Rience', 'Shani', 'Falwick of Caingorn',

#     # Important Books in the Saga
#     'The Last Wish', 'Sword of Destiny', 
#     'Blood of Elves', 'Time of Contempt', 
#     'Baptism of Fire', 'The Tower of the Swallow', 
#     'The Lady of the Lake', 'Season of Storms',

#     # Significant Locations
#     'Kaer Morhen', 'Cintra', 'Thanedd Island', 'Aretuza', 
#     'Vengerberg', 'Novigrad', 'Rivia', 'Vizima', 'Skellige Isles',
#     'Brokilon Forest', 'Gors Velen', 'Toussaint', 
#     'Nilfgaard', 'Pontar Valley', 'Redania', 'Temeria',
#     'Zerrikania', 'Mahakam', 'Brenna', 'Tir ná Lia (Aen Elle Realm)',

#     # Races and Creatures
#     'Humans', 'Elves (Aen Seidhe)', 'Dwarves', 'Halflings', 
#     'Dryads', 'Gnomes', 'Zerrikanians', 'Aen Elle Elves',
#     'Vampires (Higher and Lesser)', 'Werewolves', 'Strigas', 
#     'Djinns', 'Ghouls', 'Wraiths', 'Kikimoras', 'Leshens',
#     'Griffins', 'Basilisks', 'Sylvans', 'Chorts', 'Fiends',

#     # Organizations and Factions
#     'The Lodge of Sorceresses', 'The Brotherhood of Sorcerers',
#     'Nilfgaardian Empire', 'Scoia\'tael', 'The Northern Kingdoms',
#     'The Order of the White Rose', 'Redanian Secret Service',
#     'Temerian Special Forces', 'Aen Elle Elves',

#     # Major Plot Events and Themes
#     'The Fall of Cintra', 'The Thanedd Coup', 'The Battle of Brenna',
#     'The Witcher Trials', 'The Wild Hunt (Aen Elle)', 'The Curse of the Black Sun',
#     'The Law of Surprise', 'The White Frost Prophecy',
#     'The Conjunction of the Spheres', 'The Eternal Fire',
#     'The Search for Ciri', 'The Rescue at Stygga Castle',
#     'The Love Triangle (Geralt, Yennefer, Triss)', 'The Nilfgaardian Wars',

#     # Notable Items
#     'Aerondight', 'Witcher Medallion', 'The Rose of Shaerrawedd',
#     'Yennefer\'s Obsidian Star Necklace', 'Dandelion\'s Lute',
#     'Triss\' Amulet', 'Vilgefortz\'s Staff', 'The Book of Elves',
#     'The Broken Sword of Destiny', 'The Tower of the Swallow Portal',

#     # Magic and Sorcery
#     'The Trial of the Grasses', 'Aard', 'Igni', 'Yrden', 
#     'Quen', 'Axii', 'The Elder Speech', 
#     'Fire Magic (Forbidden)', 'Teleportation', 'Polymorphing',
#     'Glamours', 'Dimensional Portals', 'Chaos Magic',

#     # Themes and Symbolism
#     'Destiny and Free Will', 'The Nature of Monsters', 
#     'Racism and Prejudice', 'War and Politics', 
#     'Love and Sacrifice', 'The Balance Between Chaos and Order',
#     'The Fragility of Peace', 'Parenthood (Geralt and Ciri)',

#     # Side Characters and Notable Figures
#     'Eist Tuirseach', 'Pavetta', 'Calanthe', 
#     'Mousesack (Ermion)', 'Vizimir II of Redania', 
#     'Foltest of Temeria', 'Adda the White', 
#     'Hjalmar an Craite', 'Crach an Craite', 
#     'Detmold', 'Yarpen Zigrin', 'Istredd', 
#     'Vysogota of Corvo', 'Tor Lara', 'Corinne Tilly',

#     # Significant Quotes
#     '"People like to invent monsters and monstrosities. Then they seem less monstrous themselves."',
#     '"Evil is evil. Lesser, greater, middling — makes no difference."',
#     '"If I have to choose between one evil and another, I’d rather not choose at all."',
#     '"Destiny is a double-edged sword."',
#     '"There’s a grain of truth in every fairy tale."',

#     # Easter Eggs and References
#     'References to Arthurian Legends (Lady of the Lake)', 
#     'The Tale of the Three Jackdaws', 
#     'References to Eastern European Folklore', 
#     'The Swallow (Zireael) Symbolism', 'Connections to Classic Fantasy Tropes'
# ]

# witcher_universe_list = [
#     # Main Characters
#     'Geralt of Rivia', 'Ciri (Cirilla Fiona Elen Riannon)', 
#     'Yennefer of Vengerberg', 'Triss Merigold', 'Dandelion (Jaskier)', 
#     'Vesemir', 'Emhyr var Emreis', 'Vilgefortz of Roggeveen', 
#     'Regis (Emiel Regis Rohellec Terzieff-Godefroy)', 
#     'Cahir Mawr Dyffryn aep Ceallach', 'Zoltan Chivay', 
#     'Philippa Eilhart', 'Fringilla Vigo', 'Gaunter O\'Dimm (Master Mirror)', 
#     'Dettlaff van der Eretein', 'Olgierd von Everec', 
#     'Syanna (Sylvia Anna)', 'The Unseen Elder', 'Eredin Bréacc Glas',
#     'Leo Bonhart', 'The Lady of the Lake (Nimue)', 'Angoulême', 
#     'Vysogota of Corvo',

#     # Significant Locations
#     'Kaer Morhen', 'Cintra', 'Vizima', 'Novigrad', 
#     'Oxenfurt', 'Toussaint', 'Beauclair', 'Skellige Isles', 
#     'Brokilon Forest', 'Aretuza (Mage Academy)', 
#     'Thanedd Island', 'Nilfgaard', 'Pontar Valley', 
#     'Redania', 'Temeria', 'Mahakam', 'Zerrikania', 
#     'The Land of a Thousand Fables', 'Tir ná Lia (Aen Elle Realm)', 
#     'Tesham Mutna', 'Rivia',

#     # Key Races and Species
#     'Humans', 'Witchers', 'Elves (Aen Seidhe)', 'Dwarves', 
#     'Halflings', 'Gnomes', 'Dryads', 'Higher Vampires', 
#     'Lesser Vampires', 'Aen Elle Elves', 'Zerrikanians', 
#     'Werewolves', 'Strigas', 'Griffins', 'Basilisks', 
#     'Kikimoras', 'Leshens', 'Fiends', 'Scolopendromorphs', 
#     'Wraiths', 'Sylvans', 'Garkains', 'Chorts',

#     # Factions and Organizations
#     'The Lodge of Sorceresses', 'The Brotherhood of Sorcerers', 
#     'The Wild Hunt (Aen Elle)', 'The Nilfgaardian Empire', 
#     'The Northern Kingdoms', 'Scoia\'tael', 'Redanian Secret Service',
#     'Temerian Special Forces', 'Order of the Flaming Rose', 
#     'Witcher Schools (Wolf, Cat, Griffin, Bear, Viper, Manticore)',

#     # Important Events
#     'The Fall of Cintra', 'The Thanedd Coup', 
#     'The Conjunction of the Spheres', 'The Battle of Brenna', 
#     'The Nilfgaardian Wars', 'The Curse of the Black Sun',
#     'The Rescue at Stygga Castle', 'The Wild Hunt Invasion',
#     'The White Frost Prophecy',

#     # Lore Concepts
#     'The Law of Surprise', 'The Witcher Trials (Trial of the Grasses)', 
#     'Elder Blood (Hen Ichaer)', 'The Curse of the Black Sun', 
#     'The Balance Between Chaos and Order', 
#     'The Eternal Fire Religion', 'Mutations', 
#     'The Prophecy of Ithlinne', 'Dimensional Portals', 
#     'Chaos Magic', 'The Five Chivalric Virtues of Toussaint',

#     # Notable Items
#     'Aerondight (Sword of Legend)', 'Witcher Medallion', 
#     'The Rose of Shaerrawedd', 'Iris (Olgierd\'s Saber)', 
#     'Viper School Gear', 'Grandmaster Witcher Gear', 
#     'Yennefer\'s Obsidian Star', 'Dandelion\'s Lute', 
#     'The Book of Elves', 'Triss\' Amulet', 'The Alzur\'s Shield Spell',
#     'Syanna\'s Ribbon', 'The Wreath of Immortelles',

#     # Magical Spells and Signs
#     'Aard (Telekinetic Blast)', 'Igni (Fire Blast)', 
#     'Yrden (Magic Trap)', 'Quen (Protective Shield)', 
#     'Axii (Mind Control)', 'Dimensional Portals', 
#     'Fire Magic (Forbidden)', 'Polymorphing',

#     # Significant Themes
#     'Destiny vs Free Will', 'The Nature of Monsters', 
#     'Racism and Prejudice', 'War and Political Intrigue', 
#     'Love and Sacrifice', 'Parenthood (Geralt and Ciri)', 
#     'The Fragility of Peace', 'The Cost of Power', 
#     'Moral Ambiguity', 'Honor and Chivalry',

#     # Key Quests and Plotlines
#     'The Search for Ciri', 'The Wild Hunt Pursuit', 
#     'Gaunter O\'Dimm\'s Faustian Deal', 
#     'The Mystery of the Beast of Toussaint', 
#     'The Thanedd Island Coup', 'The Battle at Kaer Morhen', 
#     'The Redemption of Olgierd von Everec', 
#     'The Rescue of Yennefer from Vilgefortz',

#     # Cultural and Literary References
#     'Arthurian Legends (Lady of the Lake)', 
#     'Eastern European Folklore', 'Faustian Bargains (Gaunter O\'Dimm)',
#     'The Three Jackdaws Tale', 'The Swallow (Zireael) Symbolism',
#     'Dark Fantasy Tropes',

#     # Music and Audio
#     'The Wolven Storm (Priscilla\'s Song)', 
#     'The Trail (Main Theme)', 'Blood and Wine Main Theme', 
#     'Lullaby of Woe', 'Gaunter O\'Dimm\'s Whistling Tune',
#     'Beauclair Theme', 'Lady of the Lake Theme',

#     # Miscellaneous and Easter Eggs
#     'The Unicorn Joust in Toussaint', 'The Painted World in Hearts of Stone', 
#     'References to Andrzej Sapkowski\'s Real-Life Inspirations', 
#     'The Return of the Lady of the Lake', 'Magical Spoon Wight Quest', 
#     'Land of a Thousand Fables',
# ]


# witcher_easter_eggs_and_facts = [
#     "Geralt's bathtub meme",
#     "Unicorn joust in Toussaint",
#     "Gaunter O'Dimm's whistling tune (Devil Went Down to Georgia)",
#     "Painted world inspired by Alice in Wonderland",
#     "Roach on rooftops",
#     "Geralt quotes Yoda ('Do or do not, there is no try')",
#     "Devil’s Bridge in Kaer Morhen",
#     "Missing keys grave (Blood and Wine)",
#     "Skyrim reference in Kaer Morhen",
#     "Dandelion's lute (reference to 'The Ballad of the Lute')",
#     "Witcher 3 reveal trailer foreshadowing",
#     "Meme 'I’m not your father' line",
#     "Unseen Elder inspired by 'The Thing'",
#     "Triss' necklace symbol (alchemy)",
#     "Witcher medallion connection to real-world mythology",
#     "Dandelion and Shakespeare",
#     "Vampires as references to Nosferatu"
# ]

# witcher_lore_specialist_key_points = [
#     # Lore & World-Building
#     "The Conjunction of the Spheres",
#     "The Elder Blood Prophecy",
#     "The Wild Hunt (Aen Elle)",
#     "Witcher Medallion (Wolf-shaped)",
#     "The Law of Surprise",
#     "The Trial of the Grasses",
#     "Chaos and Order (Balance)",
#     "The Elder Speech (Elven Language)",
#     "Worlds Colliding (Monsters, Magic)",
#     "The White Frost Prophecy",
#     "The Witcher’s Code (Neutrality)",
    
#     # Important Locations
#     "Kaer Morhen (Witcher Keep)",
#     "Skellige Isles (Nordic Inspiration)",
#     "Toussaint (Vibrant, Chivalric Land)",
#     "Nilfgaardian Empire (South)",
#     "Aretuza (Mage Academy)",
#     "Vengerberg (Capital of Aedirn)",
#     "Brokilon Forest (Dryad Land)",
#     "Temeria (Fallen Kingdom)",
#     "Oxenfurt (University)",
#     "Redania (King Radovid’s Domain)",
#     "Mahakam (Dwarven Mountains)",
#     "Zerrikania (Eastern Desert)",
#     "Land of a Thousand Fables (Blood and Wine)",
    
#     # Iconic Characters
#     "Geralt of Rivia (The White Wolf)",
#     "Ciri (Elder Blood, Lion Cub of Cintra)",
#     "Yennefer of Vengerberg (Sorceress, Geralt’s Love)",
#     "Dandelion (Jaskier, the Bard)",
#     "Vesemir (The Old Mentor)",
#     "Vilgefortz (Powerful Sorcerer)",
#     "Regis (Higher Vampire, Ally)",
#     "Emhyr var Emreis (Emperor of Nilfgaard)",
#     "Cahir Mawr Dyffryn aep Ceallach (Nilfgaardian Commander)",
#     "Triss Merigold (Sorceress, Geralt’s Friend)",
#     "Philippa Eilhart (Sorceress, Lodge)",
#     "Fringilla Vigo (Sorceress, Nilfgaard)",
#     "Zoltan Chivay (Dwarf, Companion)",
#     "Angoulême (Skellige, Bard)",
#     "Letho of Gulet (Kingslayer)",
#     "Dethmold (Sorcerer)",
#     "Dettlaff van der Eretein (Higher Vampire, Blood and Wine)",
#     "Olgierd von Everec (Blood and Wine)",
#     "Syanna (Blood and Wine)",
#     "The Unseen Elder (Higher Vampire, Wild Hunt)",
#     "Eredin Bréacc Glas (Leader of the Wild Hunt)",
    
#     # Important Factions & Organizations
#     "The Lodge of Sorceresses",
#     "The Brotherhood of Sorcerers",
#     "The School of the Wolf (Kaer Morhen)",
#     "The School of the Cat (Famous for Assassin Witchers)",
#     "The School of the Bear",
#     "The School of the Griffin",
#     "The School of the Viper",
#     "The Wild Hunt (Aen Elle)",
#     "The Nilfgaardian Empire",
#     "The Northern Kingdoms",
#     "The Scoia'tael (Elf Freedom Fighters)",
#     "The Order of the Flaming Rose (Religious Order)",
#     "Redanian Intelligence (Spy Network)",
    
#     # Key Events
#     "The Conjunction of the Spheres (Monsters Enter the World)",
#     "The Fall of Cintra (Ciri’s Homeland)",
#     "The Thanedd Coup (Sorceress Uprising)",
#     "The Battle of Brenna (War with Nilfgaard)",
#     "The Nilfgaardian Wars",
#     "The Wild Hunt’s Arrival",
#     "The Curse of the Black Sun (Ciri’s Mark)",
#     "The White Frost Prophecy (End of the World)",
#     "The Battle of Kaer Morhen (Witcher Stand)",
#     "The Fall of Temeria (Political Collapse)",
    
#     # Iconic Monsters & Creatures
#     "Witcher Signs (Aard, Igni, Yrden, Quen, Axii)",
#     "Griffins (King of Beasts)",
#     "Basilisks (Stone-Creatures)",
#     "Leshens (Forest Guardians)",
#     "Fiends (Vile, Demon-Like Beasts)",
#     "Kikimoras (Monster from Swamps)",
#     "Chorts (Forest Spirits)",
#     "Wraiths (Ghostly, Vengeful Spirits)",
#     "Strigas (Cursed Princesses)",
#     "Vampires (Higher Vampires, Lesser Vampires)",
#     "Werewolves",
#     "Drowners (Water-logged Monsters)",
#     "Djinns (Genie-like, Dangerous)",
#     "Elementals (Fire, Earth, Air Spirits)",
#     "Sylvans (Forest Creatures)",
#     "Garkains (Bloodthirsty Vampires)",
    
#     # Magical Concepts & Items
#     "Elder Blood (Ciri’s Power)",
#     "The Rose of Shaerrawedd (Important Artifact)",
#     "The Witcher Medallion (Magical Vibration)",
#     "The Book of Elves (Ancient Tome)",
#     "Alzur’s Shield (Powerful Sorcery)",
#     "The Unicorn (Symbol of Love & Magic)",
#     "The Golden Dragon (Mythical Creature)",
#     "The Mirror of Gaunter O'Dimm",
#     "Witcher Potions (Swallow, White Raffard’s Decoction)",
#     "The Grandmaster Witcher Gear (High-Quality Armor and Weapons)",
    
#     # Concepts & Themes
#     "Destiny vs Free Will (Fate as a Driving Force)",
#     "Moral Ambiguity (Good vs Evil)",
#     "The Cost of Power",
#     "Monsters as Metaphors for Humanity",
#     "Parenthood (Geralt and Ciri)",
#     "Racism and Prejudice (Elves, Dwarves, and Humans)",
#     "The Law of Surprise (Destiny’s Debt)",
#     "Neutrality (Geralt's Code)",
#     "Honor and Chivalry",
#     "Love as a Redemptive Force",
#     "Sacrifice for Greater Good",
    
#     # Lesser-Known Facts & Easter Eggs
#     "The Painted World (Hearts of Stone)",
#     "Roach (Geralt’s Horse, often in odd places)",
#     "Dandelion’s Lute (Shakespearean Inspirations)",
#     "Syanna’s Ribbon (Blood and Wine)",
#     "Gaunter O'Dimm’s Whistling Tune (Devil Went Down to Georgia)",
#     "The 'I’m Not Your Father' Line (Ciri and Geralt)",
#     "Unicorn Jousting (Blood and Wine Easter Egg)",
#     "The Witcher 3 Reveal Trailer (Bathtub Scene)",
#     "Kaer Morhen’s Devil’s Bridge (Real-Life Inspiration)",
#     "Geralt’s Yoda Quote ('Do or Do Not')",
#     "Hidden Gwent Card (The Geralt Card)",
#     "The 'Missing Keys' Grave (Blood and Wine)",
#     "The Witcher’s Vampire Lore (Nosferatu and Vlad Tepes)",
#     "A Minor Reference to *Skyrim* in Kaer Morhen",
#     "Letho of Gulet (Kingslayer)",
#     "Nimue, The Lady of the Lake (Arthurian Legend)",
#     "Zireael (Swallow, Symbol of Freedom)",
# ]