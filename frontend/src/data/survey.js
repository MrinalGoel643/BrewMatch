// Survey questions and preference mapping logic

export const SURVEY_QUESTIONS = [
  {
    id: "origins",
    title: "What origins do you gravitate toward?",
    subtitle: "Select all that excite you",
    type: "multi-select",
    options: [
      { value: "Ethiopia", label: "Ethiopia", emoji: "🇪🇹", hint: "Floral, fruity, complex" },
      { value: "Colombia", label: "Colombia", emoji: "🇨🇴", hint: "Balanced, caramel, versatile" },
      { value: "Kenya", label: "Kenya", emoji: "🇰🇪", hint: "Bright, berry, wine-like" },
      { value: "Guatemala", label: "Guatemala", emoji: "🇬🇹", hint: "Chocolatey, nutty, balanced" },
      { value: "Brazil", label: "Brazil", emoji: "🇧🇷", hint: "Nutty, chocolatey, low acid" },
      { value: "Costa Rica", label: "Costa Rica", emoji: "🇨🇷", hint: "Clean, honey-sweet, bright" },
      { value: "Indonesia", label: "Indonesia", emoji: "🇮🇩", hint: "Earthy, full-bodied, spicy" },
      { value: "Rwanda", label: "Rwanda", emoji: "🇷🇼", hint: "Fruity, floral, tea-like" },
      { value: "Peru", label: "Peru", emoji: "🇵🇪", hint: "Mild, nutty, approachable" },
      { value: "Yemen", label: "Yemen", emoji: "🇾🇪", hint: "Wild, wine-like, unique" },
    ],
  },
  {
    id: "brewing",
    title: "How do you usually brew?",
    subtitle: "Select all that apply",
    type: "multi-select",
    options: [
      { value: "pourover", label: "Pour-over", emoji: "☕", hint: "V60, Chemex, Kalita" },
      { value: "espresso", label: "Espresso", emoji: "🔥", hint: "Shots, milk drinks" },
      { value: "immersion", label: "Immersion", emoji: "🫖", hint: "French press, AeroPress, Clever" },
      { value: "batch", label: "Batch brew", emoji: "🫗", hint: "Drip machine, Moccamaster" },
      { value: "coldbrew", label: "Cold brew", emoji: "🧊", hint: "Overnight steeping" },
      { value: "varied", label: "I switch it up", emoji: "🎲", hint: "Different methods for different moods" },
    ],
  },
  {
    id: "processing",
    title: "Processing methods you enjoy?",
    subtitle: "How the cherry becomes the bean",
    type: "multi-select",
    options: [
      { value: "Washed / Wet", label: "Washed", emoji: "💧", hint: "Clean, bright, origin-forward" },
      { value: "Natural / Dry", label: "Natural", emoji: "☀️", hint: "Fruity, funky, full-bodied" },
      { value: "Honey", label: "Honey", emoji: "🍯", hint: "Sweet, balanced, syrupy" },
      { value: "Anaerobic", label: "Anaerobic", emoji: "🧪", hint: "Experimental, wild, unique" },
    ],
  },
  {
    id: "notes",
    title: "Flavor notes you love?",
    subtitle: "What makes you reach for another sip",
    type: "multi-select",
    options: [
      { value: "fruity", label: "Fruity", emoji: "🍒", hint: "Berry, citrus, tropical" },
      { value: "floral", label: "Floral", emoji: "🌸", hint: "Jasmine, lavender, rose" },
      { value: "chocolatey", label: "Chocolatey", emoji: "🍫", hint: "Dark chocolate, cocoa" },
      { value: "nutty", label: "Nutty", emoji: "🥜", hint: "Almond, hazelnut, walnut" },
      { value: "caramel", label: "Caramel", emoji: "🍮", hint: "Brown sugar, toffee, honey" },
      { value: "spicy", label: "Spicy", emoji: "🌶️", hint: "Cinnamon, clove, pepper" },
      { value: "earthy", label: "Earthy", emoji: "🌍", hint: "Tobacco, leather, cedar" },
    ],
  },
  {
    id: "acidity",
    title: "How do you feel about acidity?",
    subtitle: "That bright, zingy quality",
    type: "single-select",
    options: [
      { value: "high", label: "Love it bright", emoji: "⚡", hint: "Give me that sparkle" },
      { value: "medium", label: "Keep it balanced", emoji: "⚖️", hint: "Not too much, not too little" },
      { value: "low", label: "Smooth and mellow", emoji: "🌊", hint: "Easy-drinking, round" },
    ],
  },
  {
    id: "roasters",
    title: "Any favorite roasters?",
    subtitle: "Optional — helps us understand your taste",
    type: "multi-select",
    optional: true,
    options: [
      { value: "onyx", label: "Onyx Coffee Lab" },
      { value: "counter-culture", label: "Counter Culture" },
      { value: "intelligentsia", label: "Intelligentsia" },
      { value: "stumptown", label: "Stumptown" },
      { value: "heart", label: "Heart Coffee" },
      { value: "george-howell", label: "George Howell" },
      { value: "verve", label: "Verve" },
      { value: "blue-bottle", label: "Blue Bottle" },
      { value: "sey", label: "SEY" },
      { value: "proud-mary", label: "Proud Mary" },
      { value: "other", label: "Other / local favorites" },
    ],
  },
];

// Map survey responses to taste profile for the ML model
export function surveyToPreferences(responses) {
  const { origins = [], brewing = [], processing = [], notes = [], acidity = "medium" } = responses;

  // Base scores (CQI scale: 6-10)
  let aromaScore = 8.0;
  let flavorScore = 8.0;
  let aftertasteScore = 7.5;
  let acidityScore = 8.0;
  let bodyScore = 8.0;
  let balanceScore = 8.0;
  let uniformityScore = 10.0;
  let cleanCupScore = 10.0;
  let sweetnessScore = 10.0;

  // Adjust based on acidity preference
  if (acidity === "high") {
    acidityScore = 8.5;
    bodyScore = 7.5;
  } else if (acidity === "low") {
    acidityScore = 7.0;
    bodyScore = 8.5;
  }

  // Adjust based on brew methods (now multi-select)
  const brewMethods = brewing || [];
  if (brewMethods.includes("espresso")) {
    bodyScore += 0.4;
    balanceScore += 0.2;
  }
  if (brewMethods.includes("pourover")) {
    acidityScore += 0.3;
    aromaScore += 0.3;
  }
  if (brewMethods.includes("immersion")) {
    bodyScore += 0.3;
  }
  if (brewMethods.includes("coldbrew")) {
    bodyScore += 0.2;
    acidityScore -= 0.2;
  }
  if (brewMethods.includes("batch")) {
    balanceScore += 0.2;
  }

  // Adjust based on flavor notes
  if (notes.includes("fruity") || notes.includes("floral")) {
    aromaScore += 0.3;
    acidityScore += 0.2;
  }
  if (notes.includes("chocolatey") || notes.includes("nutty")) {
    bodyScore += 0.2;
    balanceScore += 0.2;
  }
  if (notes.includes("earthy") || notes.includes("spicy")) {
    bodyScore += 0.3;
    aftertasteScore += 0.2;
  }

  // Adjust based on processing preference
  if (processing.includes("Natural / Dry")) {
    sweetnessScore = 9.8;
    bodyScore += 0.2;
  }
  if (processing.includes("Washed / Wet")) {
    cleanCupScore = 10.0;
    acidityScore += 0.1;
  }

  // Clamp all scores to valid range
  const clamp = (val) => Math.max(6.0, Math.min(10.0, val));

  return {
    aroma: clamp(aromaScore),
    flavor: clamp(flavorScore),
    aftertaste: clamp(aftertasteScore),
    acidity: clamp(acidityScore),
    body: clamp(bodyScore),
    balance: clamp(balanceScore),
    uniformity: clamp(uniformityScore),
    clean_cup: clamp(cleanCupScore),
    sweetness: clamp(sweetnessScore),
    // Pass through for roaster matching
    origins,
    processing,
    notes,
    acidityPref: acidity,
  };
}
