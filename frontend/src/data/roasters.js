// Curated specialty roaster database
// Each roaster has a flavor profile that maps to our recommendation system

export const ROASTERS = [
  {
    id: "onyx",
    name: "Onyx Coffee Lab",
    location: "Rogers, AR",
    website: "https://onyxcoffeelab.com/collections/coffee",
    style: "Competition-grade precision, clean and vibrant",
    specialties: ["Ethiopia", "Colombia", "Kenya"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "counter-culture",
    name: "Counter Culture",
    location: "Durham, NC",
    website: "https://counterculturecoffee.com/collections/coffee",
    style: "Approachable specialty, consistent quality",
    specialties: ["Ethiopia", "Colombia", "Guatemala"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "medium-high", body: "medium", notes: ["fruity", "chocolatey", "caramel"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "intelligentsia",
    name: "Intelligentsia",
    location: "Chicago, IL",
    website: "https://www.intelligentsia.com/collections/coffee",
    style: "Third-wave pioneer, direct trade focus",
    specialties: ["Ethiopia", "Kenya", "Costa Rica"],
    processing: ["Washed / Wet", "Honey"],
    flavorProfile: { acidity: "medium-high", body: "medium", notes: ["fruity", "floral", "caramel"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "stumptown",
    name: "Stumptown Coffee",
    location: "Portland, OR",
    website: "https://www.stumptowncoffee.com/collections/coffee",
    style: "Pacific Northwest classic, balanced and approachable",
    specialties: ["Guatemala", "Ethiopia", "Indonesia"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "medium", body: "medium-full", notes: ["chocolatey", "nutty", "fruity"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "heart",
    name: "Heart Coffee",
    location: "Portland, OR",
    website: "https://www.heartroasters.com/collections/beans",
    style: "Nordic-inspired, light and delicate",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "high", body: "light", notes: ["floral", "fruity"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "george-howell",
    name: "George Howell Coffee",
    location: "Boston, MA",
    website: "https://store.georgehowellcoffee.com/coffees/all/",
    style: "Legendary roaster, terroir-focused",
    specialties: ["Ethiopia", "Panama", "Kenya"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["floral", "fruity", "caramel"] },
    priceRange: "$$$$",
    shipsNationwide: true,
  },
  {
    id: "verve",
    name: "Verve Coffee",
    location: "Santa Cruz, CA",
    website: "https://www.vervecoffee.com/collections/all-coffee",
    style: "California sunshine in a cup, bright and clean",
    specialties: ["Ethiopia", "Colombia", "Costa Rica"],
    processing: ["Washed / Wet", "Honey"],
    flavorProfile: { acidity: "high", body: "medium", notes: ["fruity", "floral", "caramel"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "proud-mary",
    name: "Proud Mary",
    location: "Portland, OR / Melbourne, AU",
    website: "https://proudmarycoffee.com/collections/all-coffee",
    style: "Australian precision meets Portland creativity",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet", "Natural / Dry", "Anaerobic"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$$",
    shipsNationwide: true,
  },
  {
    id: "ruby",
    name: "Ruby Coffee",
    location: "Nelsonville, WI",
    website: "https://rubycoffeeroasters.com/collections/coffee",
    style: "Midwest gem, meticulous sourcing",
    specialties: ["Ethiopia", "Colombia", "Kenya"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "medium-high", body: "light-medium", notes: ["fruity", "floral", "caramel"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "passenger",
    name: "Passenger Coffee",
    location: "Lancaster, PA",
    website: "https://drinkpassenger.com/collections/coffee",
    style: "East coast excellence, balanced profiles",
    specialties: ["Ethiopia", "Colombia", "Guatemala"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "medium-high", body: "medium", notes: ["fruity", "chocolatey", "caramel"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "madcap",
    name: "Madcap Coffee",
    location: "Grand Rapids, MI",
    website: "https://www.madcapcoffee.com/",
    style: "Refined Midwest roasting, terroir expression",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "ritual",
    name: "Ritual Coffee",
    location: "San Francisco, CA",
    website: "https://ritualcoffee.com/collection/coffee/",
    style: "Bay Area institution, single origin focus",
    specialties: ["Ethiopia", "Guatemala", "El Salvador"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "medium-high", body: "medium", notes: ["fruity", "chocolatey"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "black-white",
    name: "Black & White Coffee",
    location: "Wake Forest, NC",
    website: "https://www.blackwhiteroasters.com/",
    style: "Competition winners, experimental processing",
    specialties: ["Ethiopia", "Colombia", "Kenya"],
    processing: ["Washed / Wet", "Natural / Dry", "Anaerobic"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$$",
    shipsNationwide: true,
  },
  {
    id: "cat-cloud",
    name: "Cat & Cloud",
    location: "Santa Cruz, CA",
    website: "https://catandcloud.com/collections/coffee",
    style: "Fun and approachable, crowd-pleasing roasts",
    specialties: ["Ethiopia", "Colombia", "Brazil"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "medium", body: "medium", notes: ["chocolatey", "fruity", "caramel"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "equator",
    name: "Equator Coffees",
    location: "San Rafael, CA",
    website: "https://www.equatorcoffees.com/collections/coffees",
    style: "Sustainable sourcing, rich and balanced",
    specialties: ["Ethiopia", "Guatemala", "Colombia"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "medium", body: "medium-full", notes: ["chocolatey", "nutty", "caramel"] },
    priceRange: "$$",
    shipsNationwide: true,
  },
  {
    id: "sightglass",
    name: "Sightglass Coffee",
    location: "San Francisco, CA",
    website: "https://sightglasscoffee.com/collections/coffee",
    style: "SF staple, balanced with character",
    specialties: ["Ethiopia", "Colombia", "Kenya"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "medium-high", body: "medium", notes: ["fruity", "chocolatey"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "tandem",
    name: "Tandem Coffee",
    location: "Portland, ME",
    website: "https://www.tandemcoffee.com/collections/all",
    style: "New England favorite, vibrant and clean",
    specialties: ["Ethiopia", "Colombia", "Kenya"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "brandywine",
    name: "Brandywine Coffee",
    location: "Wilmington, DE",
    website: "https://www.brandywinecoffeeroasters.com/collections/all-coffee-1",
    style: "Small-batch excellence, single origin gems",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet", "Natural / Dry"],
    flavorProfile: { acidity: "high", body: "light-medium", notes: ["fruity", "floral"] },
    priceRange: "$$$",
    shipsNationwide: true,
  },
  {
    id: "sey",
    name: "SEY Coffee",
    location: "Brooklyn, NY",
    website: "https://www.seycoffee.com/collections/coffee",
    style: "Nordic light roast, terroir transparency",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "high", body: "light", notes: ["floral", "fruity"] },
    priceRange: "$$$$",
    shipsNationwide: true,
  },
  {
    id: "la-cabra",
    name: "La Cabra",
    location: "Aarhus, Denmark",
    website: "https://lacabra.com/",
    style: "Scandinavian light roast masters",
    specialties: ["Ethiopia", "Kenya", "Colombia"],
    processing: ["Washed / Wet"],
    flavorProfile: { acidity: "high", body: "light", notes: ["floral", "fruity"] },
    priceRange: "$$$$",
    shipsNationwide: true,
  },
];

// Map flavor notes to roaster matching
export function matchRoasters(preferences) {
  const { origins = [], notes = [], acidity = "medium", processing = [] } = preferences;

  return ROASTERS.map((roaster) => {
    let score = 0;

    // Origin match (up to 30 points)
    if (origins.length > 0) {
      const originOverlap = roaster.specialties.filter((o) => origins.includes(o)).length;
      score += (originOverlap / Math.max(origins.length, 1)) * 30;
    } else {
      score += 15; // neutral
    }

    // Flavor note match (up to 30 points)
    if (notes.length > 0) {
      const noteOverlap = roaster.flavorProfile.notes.filter((n) => notes.includes(n)).length;
      score += (noteOverlap / Math.max(notes.length, 1)) * 30;
    } else {
      score += 15;
    }

    // Acidity preference match (up to 20 points)
    const acidityMap = { low: 1, medium: 2, high: 3 };
    const roasterAcidityMap = { low: 1, medium: 2, "medium-high": 2.5, high: 3 };
    const acidityDiff = Math.abs(
      (acidityMap[acidity] || 2) - (roasterAcidityMap[roaster.flavorProfile.acidity] || 2)
    );
    score += (1 - acidityDiff / 2) * 20;

    // Processing match (up to 20 points)
    if (processing.length > 0) {
      const processOverlap = roaster.processing.filter((p) => processing.includes(p)).length;
      score += (processOverlap / Math.max(processing.length, 1)) * 20;
    } else {
      score += 10;
    }

    return { ...roaster, matchScore: score };
  })
    .sort((a, b) => b.matchScore - a.matchScore)
    .slice(0, 5);
}
