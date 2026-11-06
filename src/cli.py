"""
Command-Line Interface for Football Player Market Value Prediction.
Allows interactive search by player name or custom hypothetical player valuation.
"""

import os
import sys
import json
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import FootballMarketValuePredictor, format_currency


def interactive_mode():
    predictor = FootballMarketValuePredictor()
    print("\n" + "=" * 70)
    print("⚽ FOOTBALL / SOCCER PLAYER MARKET VALUE PREDICTOR ⚽")
    print("=" * 70)
    print("1: Search Real Player by Name (from 26,000+ Transfermarkt dataset)")
    print("2: Custom Player Valuation Calculator (input custom stats)")
    print("q: Exit\n")

    while True:
        choice = input("Enter choice (1, 2, or q): ").strip()
        if choice.lower() in ["q", "exit", "quit"]:
            print("Exiting. Goodbye!")
            break

        if choice == "1":
            query = input("Enter player name (e.g. Mbappé, Haaland, Saka, Bellingham): ").strip()
            if not query:
                continue
            res = predictor.search_and_predict(query)
            if not res:
                print(f"[!] No player found matching '{query}'. Try another spelling or surname.")
            else:
                print("\n" + "-" * 60)
                print(f"Player: {res['player_name']} ({res['sub_position']} - {res['current_club']})")
                print(f"Age: {res['age']:.0f} | League: {'Top 5 Tier' if res['is_top_5_league'] else 'Other League'}")
                print(f"Actual Transfermarkt Value: {res['formatted_actual_value']}")
                print(f"Predicted Market Value:     {res['formatted_market_value']}")
                print(f"80% Confidence Interval:    {res['formatted_confidence_range']}")
                print(f"Valuation Tier:             {res['market_tier']}")
                print(f"Alignment:                  {res['alignment']}")
                print("Key Valuation Drivers:")
                for d in res["valuation_drivers"]:
                    print(f"  • {d}")
                print("-" * 60 + "\n")

        elif choice == "2":
            print("\n--- Custom Player Valuation Input ---")
            try:
                name = input("Player Name [Custom Star]: ").strip() or "Custom Star"
                age = float(input("Age [23]: ").strip() or "23")
                pos = input("Position (Attack / Midfield / Defender / Goalkeeper) [Attack]: ").strip() or "Attack"
                sub_pos = input("Sub-position (Centre-Forward / Right Winger / Left Winger / Central Midfield / Centre-Back) [Centre-Forward]: ").strip() or "Centre-Forward"
                top5_in = input("Plays in Top 5 League (England, Spain, Italy, Germany, France)? (y/n) [y]: ").strip().lower()
                is_top5 = 1.0 if top5_in in ["y", "yes", "1", ""] else 0.0
                rec_min = float(input("Minutes played in last 2 seasons [2400]: ").strip() or "2400")
                rec_g = float(input("Goals in last 2 seasons [18]: ").strip() or "18")
                rec_a = float(input("Assists in last 2 seasons [8]: ").strip() or "8")
                caps = float(input("Senior International Caps [15]: ").strip() or "15")
                cont_apps = float(input("Champions League / Continental Appearances [10]: ").strip() or "10")
                contract = float(input("Contract years remaining [3.0]: ").strip() or "3.0")

                custom_profile = {
                    "name": name,
                    "age": age,
                    "position": pos,
                    "sub_position": sub_pos,
                    "current_club_domestic_competition_id": "GB1" if is_top5 == 1.0 else "PO1",
                    "career_minutes": rec_min * 1.8,
                    "career_appearances": (rec_min * 1.8) / 80.0,
                    "career_goals": rec_g * 1.6,
                    "career_assists": rec_a * 1.6,
                    "career_yellow_cards": 4.0,
                    "career_red_cards": 0.0,
                    "recent_minutes": rec_min,
                    "recent_appearances": rec_min / 80.0,
                    "recent_goals": rec_g,
                    "recent_assists": rec_a,
                    "recent_yellow_cards": 2.0,
                    "recent_red_cards": 0.0,
                    "international_caps": caps,
                    "international_goals": rec_g * 0.4,
                    "continental_appearances": cont_apps,
                    "contract_years_left": contract,
                    "club_squad_size": 26.0,
                    "club_avg_age": 25.5,
                    "club_foreigners_pct": 55.0,
                    "club_national_players": 8.0,
                    "foot": "right",
                    "height_in_cm": 184.0
                }

                res = predictor.predict_player_profile(custom_profile)
                print("\n" + "=" * 60)
                print(f"VALUATION RESULT FOR: {res['player_name']}")
                print(f"Estimated Market Value:   {res['formatted_market_value']}")
                print(f"80% Confidence Interval:  {res['formatted_confidence_range']}")
                print(f"Classification Tier:      {res['market_tier']}")
                print("Key Drivers:")
                for d in res["valuation_drivers"]:
                    print(f"  • {d}")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"[!] Error processing custom input: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Football Soccer Player Market Value Predictor")
    parser.add_argument("--player", type=str, help="Search and evaluate a real football player by name")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt")
    args = parser.parse_args()

    predictor = FootballMarketValuePredictor()

    if args.player:
        res = predictor.search_and_predict(args.player)
        if not res:
            print(f"No player found matching '{args.player}'.")
        else:
            print(json.dumps(res, indent=2))
    elif args.interactive or len(sys.argv) == 1:
        interactive_mode()


if __name__ == "__main__":
    main()
