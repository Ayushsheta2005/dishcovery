"""Synthetic Yelp-format dataset for laptop smoke tests and CI.

Writes ``yelp_academic_dataset_business.json`` and
``yelp_academic_dataset_review.json`` with the real Yelp field names, so the
exact same loaders run on synthetic and real data. Businesses cluster around
neighbourhood centres inside the Yelp metros; ~12% belong to real-world chains,
~10% are non-food businesses (to exercise filtering); reviews mention dishes
the restaurant serves, with sentiment consistent with stars and occasional
misspellings ("biriyani").
"""
from __future__ import annotations

import json
import math
import os
import random
import string
from typing import Dict, List, Tuple

from dishcovery.data.cities import METROS
from dishcovery.data.geo import destination
from dishcovery.data.taxonomy import Taxonomy

CHAINS: List[Tuple[str, str, List[str], int]] = [
    # (brand, cuisine, Yelp categories, price tier)
    ("Domino's Pizza", "pizza", ["Pizza", "Fast Food"], 1),
    ("Pizza Hut", "pizza", ["Pizza", "Fast Food", "Chicken Wings"], 1),
    ("Papa John's Pizza", "pizza", ["Pizza", "Fast Food"], 1),
    ("McDonald's", "american", ["Fast Food", "Burgers"], 1),
    ("Burger King", "american", ["Fast Food", "Burgers"], 1),
    ("Wendy's", "american", ["Fast Food", "Burgers"], 1),
    ("Five Guys", "american", ["Burgers", "Fast Food"], 2),
    ("Chick-fil-A", "american", ["Fast Food", "Chicken Shop"], 1),
    ("Popeyes Louisiana Kitchen", "american", ["Fast Food", "Chicken Wings"], 1),
    ("Wingstop", "pizza", ["Chicken Wings"], 1),
    ("Subway", "sandwiches", ["Sandwiches", "Fast Food"], 1),
    ("Jersey Mike's Subs", "sandwiches", ["Sandwiches", "Delis"], 1),
    ("Chipotle Mexican Grill", "mexican", ["Mexican", "Fast Food"], 1),
    ("Taco Bell", "mexican", ["Mexican", "Fast Food", "Tex-Mex"], 1),
    ("Panda Express", "chinese", ["Chinese", "Fast Food"], 1),
    ("Starbucks", "cafe", ["Coffee & Tea", "Cafes"], 2),
    ("Dunkin'", "bakery_desserts", ["Donuts", "Coffee & Tea"], 1),
    ("Panera Bread", "sandwiches", ["Sandwiches", "Salad", "Soup"], 2),
    ("Olive Garden Italian Restaurant", "italian", ["Italian"], 2),
    ("Buffalo Wild Wings", "american", ["Chicken Wings", "Sports Bars"], 2),
]

CUISINE_WEIGHTS: Dict[str, float] = {
    "american": 12, "pizza": 9, "mexican": 9, "italian": 7, "chinese": 8, "japanese": 6, "indian": 6,
    "thai": 4, "vietnamese": 3, "korean": 2, "mediterranean": 3, "middle_eastern": 3, "breakfast_brunch": 5,
    "cafe": 6, "bakery_desserts": 4, "ice_cream": 2, "seafood": 3, "sandwiches": 5, "bbq_southern": 3,
    "salads_healthy": 2, "vegan": 1.5, "pakistani": 1, "himalayan": 1, "french": 1.5, "spanish": 1,
    "caribbean": 1.5, "cajun_creole": 1.5, "ethiopian": 1,
}

NAME_PARTS: Dict[str, Tuple[List[str], List[str]]] = {
    "indian": (["Taj", "Paradise", "Saffron", "Bombay", "Royal", "Masala", "Namaste", "Spice", "Tandoor", "Delhi"],
               ["Indian Cuisine", "Biryani House", "Kitchen", "Palace", "Grill", "Curry House", "Mahal"]),
    "pakistani": (["Karachi", "Lahori", "Shan", "Mehran", "Bundu Khan"], ["Grill", "Kabab House", "Restaurant"]),
    "himalayan": (["Everest", "Himalayan", "Yeti", "Sherpa", "Annapurna"], ["Momo House", "Kitchen", "Restaurant"]),
    "chinese": (["Golden", "Jade", "Lucky", "Red", "Great", "Panda", "Dragon", "Sichuan", "Hunan", "Canton"],
                ["Dragon", "Garden", "Palace", "Wok", "Kitchen", "House", "Express"]),
    "japanese": (["Sakura", "Tokyo", "Osaka", "Kyoto", "Zen", "Koi", "Umi", "Fuji", "Ramen"],
                 ["Sushi", "Ramen", "Sushi House", "Izakaya", "Japanese Kitchen", "Ya"]),
    "korean": (["Seoul", "Kimchi", "Gangnam", "Bulgogi", "Han"], ["Korean BBQ", "House", "Kitchen", "Garden"]),
    "thai": (["Thai", "Bangkok", "Siam", "Lotus", "Basil", "Orchid", "Chiang Mai"],
             ["Kitchen", "Cuisine", "Garden", "Thai", "Express", "Spice"]),
    "vietnamese": (["Pho", "Saigon", "Hanoi", "Little", "Golden"], ["Saigon", "Pho House", "Kitchen", "Banh Mi"]),
    "italian": (["Luigi's", "Mama Rosa's", "Bella", "Il Forno", "Tuscany", "Nonna's", "Trattoria", "Vito's"],
                ["Trattoria", "Ristorante", "Italian Kitchen", "Cucina", "Osteria"]),
    "pizza": (["Tony's", "Sal's", "Brooklyn", "Village", "Slice", "Rocco's", "Gino's", "Main Street"],
              ["Pizza", "Pizzeria", "Pizza & Grill", "Pizza Co"]),
    "american": (["Main Street", "Liberty", "Big Al's", "Corner", "Hometown", "Union", "Copper", "Blue Plate"],
                 ["Grill", "Diner", "Burgers", "Tavern", "Kitchen", "Eatery"]),
    "bbq_southern": (["Smokey", "Big Jim's", "Hickory", "Southern", "Low & Slow", "Pit"],
                     ["BBQ", "Smokehouse", "Soul Food", "Barbecue", "Kitchen"]),
    "mexican": (["El", "La", "Los", "Taqueria", "Casa", "Tres"], ["Toro", "Casita", "Cantina", "Tacos", "Jalisco", "Mexicana"]),
    "mediterranean": (["Olive", "Athens", "Santorini", "Zeus", "Aegean"], ["Grill", "Taverna", "Mediterranean Kitchen"]),
    "middle_eastern": (["Aladdin", "Beirut", "Istanbul", "Cedar", "Petra", "Halal"], ["Grill", "Kabob", "Shawarma", "Cafe"]),
    "french": (["Le", "La", "Chez", "Petit"], ["Bistro", "Brasserie", "Cafe", "Creperie"]),
    "spanish": (["Tapas", "Barcelona", "Madrid", "Sol"], ["Bar", "Tapas", "Cocina"]),
    "caribbean": (["Island", "Irie", "Jamaica", "Havana", "Caribe"], ["Grill", "Kitchen", "Jerk Hut", "Cafe"]),
    "cajun_creole": (["Bayou", "Crescent", "Big Easy", "Mardi Gras", "Cajun"], ["Kitchen", "Seafood", "Grill", "Cafe"]),
    "ethiopian": (["Addis", "Abyssinia", "Lalibela", "Queen of Sheba"], ["Ethiopian Restaurant", "Cafe", "Kitchen"]),
    "breakfast_brunch": (["Sunny Side", "Morning", "Early Bird", "Golden Yolk", "Daybreak"], ["Cafe", "Diner", "Brunch", "Kitchen"]),
    "cafe": (["Bean", "Grind", "Daily", "Third Wave", "Corner"], ["Coffee", "Cafe", "Roasters", "Espresso Bar"]),
    "bakery_desserts": (["Sweet", "Sugar", "Flour", "Butter", "Crumb"], ["Bakery", "Bakeshop", "Patisserie", "Donuts"]),
    "ice_cream": (["Scoops", "Frosty", "Polar", "Sweet Cream"], ["Creamery", "Ice Cream", "Gelato", "Froyo"]),
    "seafood": (["Harbor", "Captain's", "Ocean", "Dockside", "Pier"], ["Seafood", "Fish House", "Oyster Bar", "Crab Shack"]),
    "sandwiches": (["Primo's", "Deli", "Liberty", "Corner", "Hoagie"], ["Deli", "Subs", "Sandwich Shop", "Steaks"]),
    "salads_healthy": (["Green", "Fresh", "Harvest", "Root", "Leaf"], ["Bowl", "Kitchen", "Juice Bar", "Salads"]),
    "vegan": (["Plant", "Green", "Sprout", "Rooted", "Bloom"], ["Kitchen", "Vegan Cafe", "Eatery"]),
}

NON_FOOD = [("Nail Salon", ["Nail Salons", "Beauty & Spas"]), ("Auto Repair", ["Automotive", "Auto Repair"]),
            ("Dry Cleaners", ["Laundry Services", "Local Services"]), ("Yoga Studio", ["Fitness & Instruction"]),
            ("Hardware Store", ["Hardware Stores", "Shopping"]), ("Dental Care", ["Dentists", "Health & Medical"])]

POS = ["Amazing", "Loved the", "Best", "Fantastic", "Really good", "Delicious", "Perfectly cooked", "Must try:"]
NEG = ["Disappointing", "Cold", "Bland", "Overpriced", "Soggy", "Not great:", "Mediocre"]
FILLER = ["Service was quick.", "Portions are generous.", "Delivery took a while.", "Friendly staff.",
          "Great spot for a weeknight dinner.", "Will order again.", "Packaging was solid.", "A bit pricey.",
          "Cozy place.", "Perfect for late night cravings.", "Good vegetarian options.", "Parking is tough."]
MISSPELL = {"biryani": ["biriyani", "briyani", "biriani"], "shawarma": ["shwarma", "shawerma"],
            "chicken": ["chiken"], "margherita": ["margarita"], "gyro": ["gyros"], "tikka": ["tika"]}


def _bid(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_letters + string.digits + "-_") for _ in range(22))


def _hours(rng: random.Random, cuisine: str) -> Dict[str, str]:
    if cuisine in ("cafe", "breakfast_brunch", "bakery_desserts"):
        o, c = rng.choice([(6, 14), (7, 15), (7, 17)])
    elif rng.random() < 0.15:
        o, c = 11, 2  # late-night spot
    else:
        o, c = rng.choice([(11, 22), (11, 21), (12, 23), (10, 22), (16, 23)])
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    closed = rng.choice([None, None, "Monday", "Tuesday", "Sunday"])
    return {d: f"{o}:0-{c}:0" for d in days if d != closed}


def generate_synthetic_yelp(out_dir: str, n_businesses: int, seed: int = 0,
                            reviews_per_business: Tuple[int, int] = (3, 25),
                            taxonomy: Taxonomy = None) -> Dict[str, int]:
    rng = random.Random(seed)
    tax = taxonomy or Taxonomy.load()
    os.makedirs(out_dir, exist_ok=True)
    metros = list(METROS.values())
    weights = [m.weight for m in metros]
    hoods = {m.key: [destination(m.lat, m.lon, rng.uniform(0, 2 * math.pi), rng.uniform(0, m.radius_km * 0.7))
                     for _ in range(6)] for m in metros}
    cuisines = list(CUISINE_WEIGHTS)
    cw = [CUISINE_WEIGHTS[c] for c in cuisines]
    n_reviews = 0
    bpath = os.path.join(out_dir, "yelp_academic_dataset_business.json")
    rpath = os.path.join(out_dir, "yelp_academic_dataset_review.json")
    with open(bpath, "w") as bf, open(rpath, "w") as rf:
        for i in range(n_businesses):
            m = rng.choices(metros, weights)[0]
            hlat, hlon = rng.choice(hoods[m.key])
            lat, lon = destination(hlat, hlon, rng.uniform(0, 2 * math.pi), abs(rng.gauss(0, 1.6)))
            bid = _bid(rng)
            r = rng.random()
            if r < 0.10:
                nm, cats = rng.choice(NON_FOOD)
                name, categories, cuisine, tier = f"{rng.choice(['Elite', 'Prime', 'City', 'Star'])} {nm}", cats, None, 2
            elif r < 0.22:
                name, cuisine, cats, tier = rng.choice(CHAINS)
                categories = cats + ["Restaurants"] if cuisine not in ("cafe", "bakery_desserts") else cats + ["Food"]
            else:
                cuisine = rng.choices(cuisines, cw)[0]
                a, b = NAME_PARTS.get(cuisine, (["Main Street"], ["Kitchen"]))
                name = f"{rng.choice(a)} {rng.choice(b)}"
                ycats = tax.cuisines[cuisine].yelp_categories if cuisine in tax.cuisines else ["American (New)"]
                categories = [ycats[0]] + rng.sample(ycats[1:], min(len(ycats) - 1, rng.randint(0, 1)))
                categories += ["Food"] if cuisine in ("cafe", "bakery_desserts", "ice_cream") else ["Restaurants"]
                tier = rng.choices([1, 2, 3, 4], [0.3, 0.45, 0.2, 0.05])[0]
            stars = min(5.0, max(1.0, round(rng.gauss(3.8, 0.6) * 2) / 2))
            review_count = int(math.exp(rng.gauss(3.6, 1.1))) + 3
            biz = {"business_id": bid, "name": name, "address": f"{rng.randint(1, 9999)} {rng.choice(['Main', 'Oak', 'Market', 'Walnut', 'Pine', 'Broad'])} St",
                   "city": m.city, "state": m.state, "postal_code": f"{rng.randint(10000, 99999)}",
                   "latitude": round(lat, 7), "longitude": round(lon, 7), "stars": stars,
                   "review_count": review_count, "is_open": 1 if rng.random() > 0.05 else 0,
                   "attributes": {"RestaurantsPriceRange2": str(tier), "RestaurantsDelivery": "True"},
                   "categories": ", ".join(categories),
                   "hours": _hours(rng, cuisine or "american") if rng.random() > 0.08 else None}
            bf.write(json.dumps(biz) + "\n")
            if cuisine is None:
                continue
            # Signature dishes this place is known for: reviews concentrate on them.
            pool = tax.cuisines[cuisine].dishes if cuisine in tax.cuisines else []
            if not pool:
                continue
            signature = rng.sample(pool, min(len(pool), rng.randint(3, 6)))
            nrev = min(review_count, rng.randint(*reviews_per_business))
            for _ in range(nrev):
                rs = min(5, max(1, int(round(rng.gauss(stars, 0.9)))))
                dishes = rng.sample(signature, min(len(signature), rng.randint(1, 2)))
                parts = []
                for d in dishes:
                    dn = rng.choice([d.name] + d.variants[:2]) if rng.random() < 0.3 else d.name
                    if rng.random() < 0.06:
                        for w, alts in MISSPELL.items():
                            if w in dn:
                                dn = dn.replace(w, rng.choice(alts))
                    tag = rng.choice(d.tags) if d.tags else "tasty"
                    if rs >= 4:
                        parts.append(f"{rng.choice(POS)} {dn}! So {tag}.")
                    elif rs <= 2:
                        parts.append(f"{rng.choice(NEG)} {dn}, not {tag} at all.")
                    else:
                        parts.append(f"The {dn} was okay, a little {tag}.")
                parts.append(rng.choice(FILLER))
                rf.write(json.dumps({"review_id": _bid(rng), "user_id": _bid(rng), "business_id": bid, "stars": rs,
                                     "useful": max(0, int(rng.expovariate(0.7))), "funny": 0, "cool": 0,
                                     "text": " ".join(parts),
                                     "date": f"20{rng.randint(15, 24)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} 12:00:00"}) + "\n")
                n_reviews += 1
    return {"businesses": n_businesses, "reviews": n_reviews}
