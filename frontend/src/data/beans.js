export const BEANS = [
  {
    name: "Yirgacheffe Kochere",
    origin: "Ethiopia",
    process: "Washed / Wet",
    aroma: 8.5, flavor: 8.67, acidity: 8.42, body: 7.75, sweetness: 9.83, balance: 8.42,
    total: 88.83,
    notes: ["fruity", "floral"],
  },
  {
    name: "Huila Supremo",
    origin: "Colombia",
    process: "Washed / Wet",
    aroma: 8.42, flavor: 8.5, acidity: 8.17, body: 8.17, sweetness: 9.92, balance: 8.42,
    total: 87.67,
    notes: ["chocolatey", "caramel", "nutty"],
  },
  {
    name: "Antigua SHB",
    origin: "Guatemala",
    process: "Washed / Wet",
    aroma: 8.25, flavor: 8.33, acidity: 8.0, body: 8.25, sweetness: 9.75, balance: 8.17,
    total: 86.5,
    notes: ["chocolatey", "spicy", "earthy"],
  },
  {
    name: "Nyeri AA",
    origin: "Kenya",
    process: "Washed / Wet",
    aroma: 8.58, flavor: 8.58, acidity: 8.67, body: 7.83, sweetness: 9.67, balance: 8.33,
    total: 88.17,
    notes: ["fruity", "floral", "caramel"],
  },
  {
    name: "Cerrado Natural",
    origin: "Brazil",
    process: "Natural / Dry",
    aroma: 8.08, flavor: 8.17, acidity: 7.33, body: 8.5, sweetness: 9.83, balance: 8.08,
    total: 85.75,
    notes: ["nutty", "chocolatey", "earthy"],
  },
  {
    name: "Tarrazu SHB",
    origin: "Costa Rica",
    process: "Honey",
    aroma: 8.33, flavor: 8.42, acidity: 8.08, body: 8.0, sweetness: 9.92, balance: 8.25,
    total: 87.0,
    notes: ["caramel", "fruity", "floral"],
  },
  {
    name: "Cajamarca Organic",
    origin: "Peru",
    process: "Washed / Wet",
    aroma: 8.17, flavor: 8.25, acidity: 7.92, body: 8.08, sweetness: 9.83, balance: 8.17,
    total: 86.17,
    notes: ["chocolatey", "nutty", "caramel"],
  },
  {
    name: "Sidama Natural",
    origin: "Ethiopia",
    process: "Natural / Dry",
    aroma: 8.5, flavor: 8.58, acidity: 8.0, body: 8.17, sweetness: 9.75, balance: 8.25,
    total: 87.42,
    notes: ["fruity", "floral", "spicy"],
  },
  {
    name: "Nariño Peaberry",
    origin: "Colombia",
    process: "Washed / Wet",
    aroma: 8.33, flavor: 8.42, acidity: 8.25, body: 8.0, sweetness: 9.83, balance: 8.33,
    total: 87.25,
    notes: ["caramel", "chocolatey", "fruity"],
  },
  {
    name: "Sumatra Mandheling",
    origin: "Indonesia",
    process: "Natural / Dry",
    aroma: 8.0, flavor: 8.08, acidity: 7.25, body: 8.67, sweetness: 9.67, balance: 7.92,
    total: 84.92,
    notes: ["earthy", "spicy", "nutty"],
  },
  {
    name: "Gesha Village",
    origin: "Ethiopia",
    process: "Washed / Wet",
    aroma: 8.83, flavor: 8.92, acidity: 8.75, body: 7.58, sweetness: 9.83, balance: 8.67,
    total: 90.5,
    notes: ["floral", "fruity"],
  },
  {
    name: "Rwandan Bourbon",
    origin: "Rwanda",
    process: "Washed / Wet",
    aroma: 8.42, flavor: 8.5, acidity: 8.33, body: 8.0, sweetness: 9.75, balance: 8.25,
    total: 87.08,
    notes: ["fruity", "caramel"],
  },
];

export const FLAVOR_OPTIONS = ["Fruity", "Chocolatey", "Nutty", "Floral", "Spicy", "Earthy", "Caramel"];
export const PROCESS_OPTIONS = ["Any", "Washed / Wet", "Natural / Dry", "Honey"];
export const ORIGIN_OPTIONS = [
  "Anywhere", "Ethiopia", "Colombia", "Guatemala", "Kenya",
  "Brazil", "Costa Rica", "Peru", "Indonesia", "Rwanda",
];

export function scoreBean(bean, prefs) {
  let score = 0;
  const flavors = prefs.flavors.map((f) => f.toLowerCase());

  // flavor overlap — 30 pts
  if (flavors.length > 0) {
    const overlap = bean.notes.filter((n) => flavors.includes(n)).length;
    score += (overlap / flavors.length) * 30;
  } else {
    score += 15;
  }

  // cupping dimension proximity — 20 pts each
  score += (1 - Math.abs(bean.acidity - prefs.acidity) / 10) * 20;
  score += (1 - Math.abs(bean.body - prefs.body) / 10) * 20;
  score += (1 - Math.abs(bean.sweetness - prefs.sweetness) / 10) * 20;

  // categorical bonuses — 5 pts each
  if (prefs.process === "Any" || bean.process === prefs.process) score += 5;
  if (prefs.origin === "Anywhere" || bean.origin === prefs.origin) score += 5;

  return score;
}
