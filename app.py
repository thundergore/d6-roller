import streamlit as st
import numpy as np
from collections import Counter
from statistics import median, mode, StatisticsError
from dataclasses import dataclass
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import re

# Page config
st.set_page_config(
    page_title="D6 Roller & Combat Resolver",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .stat-container {
        padding: 10px;
        border-left: 3px solid #4CAF50;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MIN_DICE = 1
MAX_DICE = 1000
DICE_MIN = 1
DICE_MAX = 6


# Dice Expression Utilities
def parse_dice_expression(expr: str) -> Union[int, str]:
    """Parse and validate a dice expression. Returns the expression if valid, raises ValueError if not."""
    expr = expr.strip().upper().replace(" ", "")

    # Check if it's a plain integer
    if re.fullmatch(r"\d+", expr):
        return int(expr)

    # Check if it's a valid dice expression
    # Supports: D3, D6, 2D3, 2D6, D3+1, D6+3, 2D6+1, etc.
    if re.fullmatch(r"(\d+)?D[36](\+\d+)?", expr):
        return expr

    raise ValueError(f"Invalid dice expression: {expr}")


def expected_value_from_dice(expr: Union[int, str]) -> float:
    """Calculate expected value from a dice expression"""
    if isinstance(expr, int):
        return float(expr)

    expr = str(expr).strip().upper().replace(" ", "")

    # Parse expressions like "2D6+3", "D3", etc.
    match = re.fullmatch(r"(\d+)?D([36])(\+(\d+))?", expr)
    if not match:
        raise ValueError(f"Invalid dice expression: {expr}")

    num_dice = int(match.group(1)) if match.group(1) else 1
    dice_type = int(match.group(2))  # 3 or 6
    modifier = int(match.group(4)) if match.group(4) else 0

    # Expected value of dN is (N+1)/2
    expected_per_die = (dice_type + 1) / 2.0
    return num_dice * expected_per_die + modifier


def roll_dice_expression(expr: Union[int, str], rng=None) -> int:
    """Roll a dice expression and return the result"""
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(expr, int):
        return expr

    expr = str(expr).strip().upper().replace(" ", "")

    # Parse expressions like "2D6+3", "D3", etc.
    match = re.fullmatch(r"(\d+)?D([36])(\+(\d+))?", expr)
    if not match:
        raise ValueError(f"Invalid dice expression: {expr}")

    num_dice = int(match.group(1)) if match.group(1) else 1
    dice_type = int(match.group(2))  # 3 or 6
    modifier = int(match.group(4)) if match.group(4) else 0

    # Roll the dice
    rolls = rng.integers(1, dice_type + 1, size=num_dice)
    return int(np.sum(rolls)) + modifier


@dataclass
class RollResult:
    """Result of a dice roll with metadata"""
    dice: np.ndarray
    successes: np.ndarray
    failures: np.ndarray
    num_successes: int
    num_failures: int
    target: int
    modifier: int
    rerolled_indices: List[int]


@dataclass
class CombatConfig:
    """Configuration for combat resolution"""
    weapon_name: str = "Weapon"
    num_attacks: Union[int, str] = 10  # Can be int or dice expression like "2D6"
    hit_target: int = 4
    hit_modifier: int = 0
    hit_reroll_ones: bool = False
    hit_reroll_fails: bool = False
    hit_crit_value: int = 6
    hit_crits_2_hits: bool = False  # Crits count as 2 hits
    hit_auto_wound_on_crit: bool = False
    hit_mortal_wounds_on_crit: bool = False
    hit_mortals_continue: bool = False
    wound_target: int = 4
    wound_modifier: int = 0
    wound_reroll_ones: bool = False
    wound_reroll_fails: bool = False
    wound_crit_value: int = 6
    wound_mortal_wounds_on_crit: bool = False
    save_target: int = 4
    save_modifier: int = 0
    ward_target: Optional[int] = None
    ward_modifier: int = 0
    damage: Union[int, str] = 1  # Can be int or dice expression like "D3", "D6", "D3+3"
    damage_modifier: int = 0


@dataclass
class CombatResult:
    """Result of full combat resolution"""
    weapon_name: str
    hit_roll: RollResult
    wound_roll: RollResult
    save_roll: RollResult
    ward_roll: Optional[RollResult]
    normal_damage: int
    normal_damage_rolls: List[int]  # Individual damage rolls for normal damage
    mortal_wounds: int
    mortal_damage_rolls: List[int]  # Individual damage rolls for mortal wounds
    mortal_ward_roll: Optional[RollResult]
    total_damage: int
    expected_damage: float


@dataclass
class MultiWeaponResult:
    """Aggregated result from multiple weapons"""
    weapon_results: List[CombatResult]
    total_normal_damage: int
    total_mortal_wounds: int
    total_damage: int
    total_expected_damage: float


class DiceRoller:
    """Handles dice rolling logic and statistics"""

    def __init__(self):
        self.rng = np.random.default_rng()

    def roll(self, num_dice):
        """Roll the specified number of d6 dice"""
        if num_dice < MIN_DICE or num_dice > MAX_DICE:
            raise ValueError(f"Number of dice must be between {MIN_DICE} and {MAX_DICE}")
        return self.rng.integers(DICE_MIN, DICE_MAX + 1, size=num_dice)

    def roll_with_target(self, num_dice, target, modifier=0, reroll_ones=False,
                         reroll_all_fails=False, reroll_specific=None):
        """Roll dice and check against target with modifiers and rerolls"""
        if num_dice == 0:
            return RollResult(
                dice=np.array([]), successes=np.array([]), failures=np.array([]),
                num_successes=0, num_failures=0, target=target, modifier=modifier, rerolled_indices=[]
            )

        if num_dice < MIN_DICE or num_dice > MAX_DICE:
            raise ValueError(f"Number of dice must be between {MIN_DICE} and {MAX_DICE}")

        dice = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=num_dice)
        rerolled_indices = []

        if reroll_ones:
            ones_mask = dice == 1
            reroll_indices = np.where(ones_mask)[0]
            dice[ones_mask] = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=np.sum(ones_mask))
            rerolled_indices.extend(reroll_indices.tolist())

        if reroll_specific:
            reroll_mask = np.isin(dice, reroll_specific)
            reroll_indices = np.where(reroll_mask)[0]
            dice[reroll_mask] = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=np.sum(reroll_mask))
            rerolled_indices.extend(reroll_indices.tolist())

        modified_dice = dice + modifier

        if reroll_all_fails and not reroll_ones and not reroll_specific:
            fail_mask = modified_dice < target
            reroll_indices = np.where(fail_mask)[0]
            dice[fail_mask] = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=np.sum(fail_mask))
            modified_dice = dice + modifier
            rerolled_indices.extend(reroll_indices.tolist())

        success_mask = modified_dice >= target
        successes = dice[success_mask]
        failures = dice[~success_mask]

        return RollResult(
            dice=dice, successes=successes, failures=failures,
            num_successes=np.sum(success_mask), num_failures=np.sum(~success_mask),
            target=target, modifier=modifier, rerolled_indices=list(set(rerolled_indices))
        )

    def get_statistics(self, dice_result):
        """Calculate statistics for a dice roll"""
        if len(dice_result) == 0:
            return {}

        stats = {
            'mean': round(np.mean(dice_result), 3),
            'median': median(dice_result),
            'min': int(np.min(dice_result)),
            'max': int(np.max(dice_result)),
            'sum': int(np.sum(dice_result))
        }

        try:
            stats['mode'] = mode(dice_result)
        except StatisticsError:
            stats['mode'] = None

        return stats

    def get_distribution(self, dice_result):
        """Get the distribution of dice values"""
        counter = Counter(dice_result)
        return sorted(counter.items())


class CombatResolver:
    """Handles multi-stage combat resolution"""

    def __init__(self):
        self.dice_roller = DiceRoller()

    def calculate_expected_damage(self, config: CombatConfig) -> float:
        """Calculate theoretical expected damage based on probabilities"""
        # Calculate expected attacks
        expected_attacks = expected_value_from_dice(config.num_attacks)

        # Calculate expected damage per hit
        expected_damage_per_hit = expected_value_from_dice(config.damage) + config.damage_modifier

        # Calculate hit probability
        base_hit_prob = max(0, min(1, (7 - config.hit_target + config.hit_modifier) / 6))

        if config.hit_reroll_ones:
            prob_roll_one = 1/6
            hit_prob = base_hit_prob + (prob_roll_one * base_hit_prob)
        elif config.hit_reroll_fails:
            hit_prob = base_hit_prob + ((1 - base_hit_prob) * base_hit_prob)
        else:
            hit_prob = base_hit_prob

        # Handle critical hits
        prob_natural_crit = max(0, (7 - config.hit_crit_value) / 6)
        mortal_wound_damage = 0
        normal_hit_prob = hit_prob
        crit_bonus_hits = 0  # Extra hits from crits

        # Handle crits as 2 hits (must be processed first)
        if config.hit_crits_2_hits and not config.hit_mortal_wounds_on_crit and not config.hit_auto_wound_on_crit:
            # Each crit that hits generates 1 additional hit
            # Probability of rolling a crit value (e.g., 6): 1/6 or (7 - crit_value) / 6
            # Probability that a crit hits: always 1 (if crit_value >= target, which it usually is)
            if config.hit_crit_value >= config.hit_target - config.hit_modifier:
                crit_bonus_hits = prob_natural_crit * expected_attacks
            # These bonus hits go through normal wound/save/ward sequence

        if config.hit_mortal_wounds_on_crit:
            prob_crit_hits = prob_natural_crit

            if not config.hit_mortals_continue:
                # Mortals stop the attack (default behavior)
                normal_hit_prob = hit_prob - prob_crit_hits
            # If hit_mortals_continue is True, normal_hit_prob stays unchanged

            if config.ward_target:
                effective_ward_target = config.ward_target - config.ward_modifier
                ward_success_prob_mw = max(0, min(1, (7 - effective_ward_target) / 6))
                ward_fail_prob_mw = 1 - ward_success_prob_mw
                mortal_wound_damage = expected_attacks * prob_crit_hits * ward_fail_prob_mw * expected_damage_per_hit
            else:
                mortal_wound_damage = expected_attacks * prob_crit_hits * expected_damage_per_hit

        elif config.hit_auto_wound_on_crit:
            prob_crit_hits = prob_natural_crit
            auto_wound_damage_contribution = prob_crit_hits
            normal_hit_prob = hit_prob - prob_crit_hits
        else:
            auto_wound_damage_contribution = 0

        # Calculate wound probability
        base_wound_prob = max(0, min(1, (7 - config.wound_target + config.wound_modifier) / 6))

        if config.wound_reroll_ones:
            prob_roll_one = 1/6
            wound_prob = base_wound_prob + (prob_roll_one * base_wound_prob)
        elif config.wound_reroll_fails:
            wound_prob = base_wound_prob + ((1 - base_wound_prob) * base_wound_prob)
        else:
            wound_prob = base_wound_prob

        # Handle mortal wounds on wound crits
        prob_wound_crit = max(0, (7 - config.wound_crit_value) / 6)
        wound_mortal_damage = 0

        if config.wound_mortal_wounds_on_crit:
            # Expected wound rolls that will crit = normal_hit_prob * prob_wound_crit
            # These become mortals instead of normal wounds
            expected_wound_mortals_per_attack = normal_hit_prob * prob_wound_crit

            # Reduce normal wound success rate by crit rate (they become mortals instead)
            wound_prob = max(0, wound_prob - prob_wound_crit)

            # Calculate mortal damage from wound crits (subject to ward only)
            if config.ward_target:
                effective_ward_target = config.ward_target - config.ward_modifier
                ward_success_prob = max(0, min(1, (7 - effective_ward_target) / 6))
                ward_fail_prob = 1 - ward_success_prob
                wound_mortal_damage = expected_attacks * expected_wound_mortals_per_attack * ward_fail_prob * expected_damage_per_hit
            else:
                wound_mortal_damage = expected_attacks * expected_wound_mortals_per_attack * expected_damage_per_hit

        # Calculate save probability
        effective_save_target = config.save_target - config.save_modifier
        save_success_prob = max(0, min(1, (7 - effective_save_target) / 6))
        save_fail_prob = 1 - save_success_prob

        # Calculate ward probability
        if config.ward_target:
            effective_ward_target = config.ward_target - config.ward_modifier
            ward_success_prob = max(0, min(1, (7 - effective_ward_target) / 6))
            ward_fail_prob = 1 - ward_success_prob
        else:
            ward_fail_prob = 1

        # Expected normal damage
        if config.hit_auto_wound_on_crit:
            normal_damage_contribution = (normal_hit_prob * wound_prob)
            auto_wound_contribution = auto_wound_damage_contribution
            expected_normal_damage = expected_attacks * (normal_damage_contribution + auto_wound_contribution) * save_fail_prob * ward_fail_prob * expected_damage_per_hit
        else:
            expected_normal_damage = expected_attacks * normal_hit_prob * wound_prob * save_fail_prob * ward_fail_prob * expected_damage_per_hit

        # Add bonus damage from crits (these go through normal wound/save/ward)
        crit_bonus_damage = crit_bonus_hits * wound_prob * save_fail_prob * ward_fail_prob * expected_damage_per_hit

        expected_total = expected_normal_damage + crit_bonus_damage + mortal_wound_damage + wound_mortal_damage
        return round(expected_total, 2)

    def resolve_combat(self, config: CombatConfig) -> CombatResult:
        """Resolve full combat sequence"""
        # Roll for variable attacks if needed
        actual_num_attacks = roll_dice_expression(config.num_attacks, self.dice_roller.rng)

        # Stage 1: Hit rolls
        hit_roll = self.dice_roller.roll_with_target(
            actual_num_attacks, config.hit_target, config.hit_modifier,
            config.hit_reroll_ones, config.hit_reroll_fails
        )

        # Check for critical hits
        natural_crits_that_hit = 0
        mortal_wound_count = 0
        auto_wound_count = 0
        normal_hits = hit_roll.num_successes

        if len(hit_roll.successes) > 0:
            natural_crits_that_hit = np.sum(hit_roll.successes >= config.hit_crit_value)

            # Handle crits as 2 hits (must be processed first, before other crit effects)
            if config.hit_crits_2_hits:
                # Each crit generates an additional hit
                normal_hits += natural_crits_that_hit

            if config.hit_mortal_wounds_on_crit:
                mortal_wound_count = natural_crits_that_hit
                if not config.hit_mortals_continue:
                    # Mortals stop the attack sequence (default behavior)
                    normal_hits -= natural_crits_that_hit
                # If hit_mortals_continue is True, mortals are dealt but attack continues
            elif config.hit_auto_wound_on_crit:
                auto_wound_count = natural_crits_that_hit
                normal_hits -= natural_crits_that_hit

        # Stage 2: Wound rolls
        wound_roll = self.dice_roller.roll_with_target(
            normal_hits, config.wound_target, config.wound_modifier,
            config.wound_reroll_ones, config.wound_reroll_fails
        )

        # Check for critical wounds (mortal wounds on wound crits)
        wound_mortal_count = 0
        successful_wounds = wound_roll.num_successes

        if config.wound_mortal_wounds_on_crit and len(wound_roll.successes) > 0:
            wound_crits = np.sum(wound_roll.successes >= config.wound_crit_value)
            wound_mortal_count = wound_crits
            successful_wounds -= wound_crits  # These become mortals instead of normal wounds

        total_wounds = successful_wounds + auto_wound_count

        # Stage 3: Save rolls
        save_roll = self.dice_roller.roll_with_target(
            total_wounds, config.save_target, config.save_modifier
        )

        failed_saves = save_roll.num_failures

        # Stage 4: Roll damage for failed saves, then ward saves against damage points
        ward_roll = None
        normal_damage = 0
        normal_damage_rolls = []

        # First, roll damage for each failed save to generate damage points
        damage_points = 0
        for _ in range(failed_saves):
            dmg_roll = roll_dice_expression(config.damage, self.dice_roller.rng)
            total_dmg = dmg_roll + config.damage_modifier
            normal_damage_rolls.append(total_dmg)
            damage_points += total_dmg

        # Then, if ward save exists, roll against each damage point
        if config.ward_target is not None and damage_points > 0:
            ward_roll = self.dice_roller.roll_with_target(
                damage_points, config.ward_target, config.ward_modifier
            )
            normal_damage = ward_roll.num_failures
        else:
            normal_damage = damage_points

        # Stage 5: Roll damage for mortal wounds, then ward saves against damage points
        mortal_ward_roll = None
        total_mortal_count = mortal_wound_count + wound_mortal_count
        final_mortal_wounds = 0
        mortal_damage_rolls = []

        # First, roll damage for each mortal wound to generate damage points
        mortal_damage_points = 0
        for _ in range(total_mortal_count):
            dmg_roll = roll_dice_expression(config.damage, self.dice_roller.rng)
            total_dmg = dmg_roll + config.damage_modifier
            mortal_damage_rolls.append(total_dmg)
            mortal_damage_points += total_dmg

        # Then, if ward save exists, roll against each mortal damage point
        if config.ward_target is not None and mortal_damage_points > 0:
            mortal_ward_roll = self.dice_roller.roll_with_target(
                mortal_damage_points, config.ward_target, config.ward_modifier
            )
            final_mortal_wounds = mortal_ward_roll.num_failures
        else:
            final_mortal_wounds = mortal_damage_points

        total_damage = normal_damage + final_mortal_wounds
        expected_damage = self.calculate_expected_damage(config)

        return CombatResult(
            weapon_name=config.weapon_name,
            hit_roll=hit_roll, wound_roll=wound_roll, save_roll=save_roll,
            ward_roll=ward_roll, normal_damage=normal_damage,
            normal_damage_rolls=normal_damage_rolls,
            mortal_wounds=final_mortal_wounds,
            mortal_damage_rolls=mortal_damage_rolls,
            mortal_ward_roll=mortal_ward_roll,
            total_damage=total_damage, expected_damage=expected_damage
        )

    def resolve_multi_weapon_combat(self, configs: List[CombatConfig]) -> MultiWeaponResult:
        """Resolve combat with multiple weapons and aggregate results"""
        weapon_results = []
        total_normal = 0
        total_mortal = 0
        total_expected = 0.0

        for config in configs:
            result = self.resolve_combat(config)
            weapon_results.append(result)
            total_normal += result.normal_damage
            total_mortal += result.mortal_wounds
            total_expected += result.expected_damage

        return MultiWeaponResult(
            weapon_results=weapon_results,
            total_normal_damage=total_normal,
            total_mortal_wounds=total_mortal,
            total_damage=total_normal + total_mortal,
            total_expected_damage=total_expected
        )


# Initialize session state
if 'combat_history' not in st.session_state:
    st.session_state.combat_history = []
if 'roll_history' not in st.session_state:
    st.session_state.roll_history = []
if 'num_weapons' not in st.session_state:
    st.session_state.num_weapons = 1


def plot_convergence(combat_history):
    """Create a visualization showing convergence to expected values"""
    if len(combat_history) < 2:
        return None

    combats = list(range(1, len(combat_history) + 1))
    cumulative_actual = []
    cumulative_expected = []

    total_actual = 0
    total_expected = 0

    for entry in combat_history:
        total_actual += entry['actual']
        total_expected += entry['expected']
        cumulative_actual.append(total_actual)
        cumulative_expected.append(total_expected)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Top plot: Cumulative damage
    ax1.plot(combats, cumulative_actual, label='Actual Damage', marker='o', linewidth=2, markersize=6)
    ax1.plot(combats, cumulative_expected, label='Expected Damage', marker='s', linewidth=2, markersize=6, linestyle='--')
    ax1.fill_between(combats, cumulative_actual, cumulative_expected, alpha=0.2)
    ax1.set_xlabel('Combat Number', fontsize=12)
    ax1.set_ylabel('Cumulative Damage', fontsize=12)
    ax1.set_title('Damage Convergence Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Deviation percentage
    deviation_pct = [(cumulative_actual[i] - cumulative_expected[i]) / cumulative_expected[i] * 100
                     if cumulative_expected[i] > 0 else 0
                     for i in range(len(combats))]

    colors = ['green' if abs(d) < 5 else 'orange' if abs(d) < 10 else 'red' for d in deviation_pct]
    ax2.bar(combats, deviation_pct, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-5, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Combat Number', fontsize=12)
    ax2.set_ylabel('Deviation %', fontsize=12)
    ax2.set_title('Deviation from Expected (%)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def render_weapon_config(weapon_num: int) -> CombatConfig:
    """Render UI for a single weapon configuration and return the config"""
    key_suffix = f"_w{weapon_num}"

    # Weapon name
    weapon_name = st.text_input("Weapon Name", value=f"Weapon {weapon_num}", key=f"weapon_name{key_suffix}")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_attacks_input = st.text_input("Number of Attacks", value="10",
                                          help="Enter a number or dice expression (D3, D6, 2D6)",
                                          key=f"attacks{key_suffix}")
        try:
            num_attacks = parse_dice_expression(num_attacks_input)
        except ValueError as e:
            st.error(f"Invalid attacks expression: {e}")
            num_attacks = 10
    with col2:
        damage_input = st.text_input("Damage", value="1",
                                     help="Enter a number or dice expression (D3, D6, D3+1, D3+3)",
                                     key=f"damage{key_suffix}")
        try:
            damage = parse_dice_expression(damage_input)
        except ValueError as e:
            st.error(f"Invalid damage expression: {e}")
            damage = 1
    with col3:
        damage_modifier = st.selectbox("Damage Modifier", options=[-2, -1, 0, 1, 2, 3],
                                      index=2, key=f"damage_mod{key_suffix}")

    st.markdown("---")
    st.subheader("TO HIT")
    col1, col2 = st.columns(2)
    with col1:
        hit_target = st.selectbox("Hit on", options=[2, 3, 4, 5, 6], index=1,
                                 key=f"hit_target{key_suffix}")
        hit_modifier = st.selectbox("Hit Modifier", options=[-2, -1, 0, 1, 2], index=2,
                                   key=f"hit_mod{key_suffix}")
    with col2:
        hit_reroll_ones = st.checkbox("Reroll 1s", key=f"hit_reroll_1s{key_suffix}")
        hit_reroll_fails = st.checkbox("Reroll all fails", key=f"hit_reroll_fails{key_suffix}")

    col1, col2 = st.columns(2)
    with col1:
        hit_crit_value = st.selectbox("Critical Hit Value", options=[5, 6], index=1,
                                     key=f"hit_crit{key_suffix}")
    with col2:
        st.write("")  # spacing

    hit_crits_2_hits = st.checkbox(f"{hit_crit_value}+ count as 2 hits (crits)",
                                   key=f"hit_crits_2hits{key_suffix}")

    col1, col2 = st.columns(2)
    with col1:
        hit_auto_wound = st.checkbox(f"{hit_crit_value}+ auto-wound (skip wound roll)",
                                    key=f"hit_auto_wound{key_suffix}")
    with col2:
        hit_mortals = st.checkbox(f"{hit_crit_value}+ = mortal wounds",
                                 key=f"hit_mortals{key_suffix}")

    if hit_mortals:
        hit_mortals_continue = st.checkbox("Mortal wounds continue to normal damage (rare)",
                                          value=False, key=f"hit_mortals_continue{key_suffix}")
    else:
        hit_mortals_continue = False

    st.markdown("---")
    st.subheader("TO WOUND")
    col1, col2 = st.columns(2)
    with col1:
        wound_target = st.selectbox("Wound on", options=[2, 3, 4, 5, 6], index=2,
                                   key=f"wound_target{key_suffix}")
        wound_modifier = st.selectbox("Wound Modifier", options=[-2, -1, 0, 1, 2], index=2,
                                     key=f"wound_mod{key_suffix}")
    with col2:
        wound_reroll_ones = st.checkbox("Reroll 1s", key=f"wound_reroll_1s{key_suffix}")
        wound_reroll_fails = st.checkbox("Reroll all fails", key=f"wound_reroll_fails{key_suffix}")

    col1, col2 = st.columns(2)
    with col1:
        wound_crit_value = st.selectbox("Critical Wound Value", options=[5, 6], index=1,
                                       key=f"wound_crit{key_suffix}")
    with col2:
        wound_mortals = st.checkbox(f"{wound_crit_value}+ = mortal wounds",
                                   key=f"wound_mortals{key_suffix}")

    st.markdown("---")
    st.subheader("SAVES (Defender)")
    col1, col2 = st.columns(2)
    with col1:
        save_target = st.selectbox("Save on", options=[2, 3, 4, 5, 6], index=2,
                                  key=f"save_target{key_suffix}")
    with col2:
        save_modifier = st.selectbox("Save Modifier", options=[-2, -1, 0, 1, 2], index=2,
                                    key=f"save_mod{key_suffix}")

    st.markdown("---")
    st.subheader("WARD SAVE (Optional)")
    enable_ward = st.checkbox("Enable Ward Save", key=f"enable_ward{key_suffix}")
    if enable_ward:
        ward_target = st.selectbox("Ward on", options=[2, 3, 4, 5, 6], index=3,
                                  key=f"ward_target{key_suffix}")
    else:
        ward_target = None

    return CombatConfig(
        weapon_name=weapon_name,
        num_attacks=num_attacks,
        hit_target=hit_target,
        hit_modifier=hit_modifier,
        hit_reroll_ones=hit_reroll_ones,
        hit_reroll_fails=hit_reroll_fails,
        hit_crit_value=hit_crit_value,
        hit_crits_2_hits=hit_crits_2_hits,
        hit_auto_wound_on_crit=hit_auto_wound,
        hit_mortal_wounds_on_crit=hit_mortals,
        hit_mortals_continue=hit_mortals_continue,
        wound_target=wound_target,
        wound_modifier=wound_modifier,
        wound_reroll_ones=wound_reroll_ones,
        wound_reroll_fails=wound_reroll_fails,
        wound_crit_value=wound_crit_value,
        wound_mortal_wounds_on_crit=wound_mortals,
        save_target=save_target,
        save_modifier=save_modifier,
        ward_target=ward_target,
        ward_modifier=0,
        damage=damage,
        damage_modifier=damage_modifier
    )


def main():
    # Sidebar
    with st.sidebar:
        st.title("🎲 D6 Roller")
        st.markdown("---")

        st.subheader("About")
        st.write("A powerful dice roller and wargaming combat resolver with statistical analysis.")

        st.markdown("---")
        st.subheader("Features")
        st.write("✓ Simple dice rolling")
        st.write("✓ Combat resolution")
        st.write("✓ Rerolls & modifiers")
        st.write("✓ Special rules (6s)")
        st.write("✓ Statistical analysis")
        st.write("✓ Convergence tracking")

    # Main content
    st.title("🎲 D6 Roller & Combat Resolver")

    # Tabs
    tab1, tab2 = st.tabs(["🎲 Simple Rolls", "⚔️ Combat Resolver"])

    # Simple Rolls Tab
    with tab1:
        st.header("Simple Dice Roller")

        col1, col2 = st.columns([2, 1])

        with col1:
            num_dice = st.number_input("Number of d6s", min_value=1, max_value=1000, value=10, step=1)

        with col2:
            st.write("")
            st.write("")
            quick_col1, quick_col2, quick_col3 = st.columns(3)
            with quick_col1:
                quick_5 = st.button("5d6", use_container_width=True)
            with quick_col2:
                quick_10 = st.button("10d6", use_container_width=True)
            with quick_col3:
                quick_20 = st.button("20d6", use_container_width=True)

        roll_button = st.button("🎲 Roll Dice", type="primary", use_container_width=True)

        # Determine which button was pressed and what to roll
        roll_triggered = False
        dice_to_roll = num_dice

        if quick_5:
            dice_to_roll = 5
            roll_triggered = True
        elif quick_10:
            dice_to_roll = 10
            roll_triggered = True
        elif quick_20:
            dice_to_roll = 20
            roll_triggered = True
        elif roll_button:
            roll_triggered = True

        if roll_triggered:
            roller = DiceRoller()
            dice_result = roller.roll(dice_to_roll)
            stats = roller.get_statistics(dice_result)
            distribution = roller.get_distribution(dice_result)

            st.session_state.roll_history.append(dice_result)

            st.markdown("---")
            st.subheader(f"Results: Rolling {dice_to_roll} d6")

            # Individual rolls (if not too many)
            if dice_to_roll <= 50:
                st.write("**Individual rolls:**")
                st.code(str(list(dice_result)))

            # Distribution
            st.write("**Distribution:**")
            for value, count in distribution:
                percentage = (count / dice_to_roll) * 100
                bar = "█" * count
                st.text(f"{value}: {bar} ({count}) - {percentage:.1f}%")

            # Statistics
            st.write("**Statistics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sum", stats['sum'])
                st.metric("Mean", stats['mean'])
            with col2:
                st.metric("Median", stats['median'])
                if stats['mode']:
                    st.metric("Mode", stats['mode'])
            with col3:
                st.metric("Min", stats['min'])
                st.metric("Max", stats['max'])

            # Historic mean
            if len(st.session_state.roll_history) > 0:
                all_rolls = np.concatenate(st.session_state.roll_history)
                historic_mean = round(np.mean(all_rolls), 3)
                st.info(f"📊 Historic mean: {historic_mean} (across {len(st.session_state.roll_history)} roll{'s' if len(st.session_state.roll_history) > 1 else ''})")

        if st.button("Clear History"):
            st.session_state.roll_history = []
            st.success("History cleared!")

    # Combat Resolver Tab
    with tab2:
        st.header("Combat Resolver")

        # Weapon management buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**Weapons configured: {st.session_state.num_weapons}**")
        with col2:
            if st.button("➕ Add Weapon", use_container_width=True):
                st.session_state.num_weapons += 1
                st.rerun()
        with col3:
            if st.button("➖ Remove Weapon", use_container_width=True, disabled=(st.session_state.num_weapons <= 1)):
                st.session_state.num_weapons = max(1, st.session_state.num_weapons - 1)
                st.rerun()

        # Collect weapon configurations
        weapon_configs = []
        for i in range(st.session_state.num_weapons):
            with st.expander(f"⚔️ Weapon {i+1} Configuration", expanded=(i==0)):
                config = render_weapon_config(i+1)
                weapon_configs.append(config)

        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            resolve_button = st.button("⚔️ Resolve Combat", type="primary", use_container_width=True)
        with col2:
            if st.button("Clear Statistics", use_container_width=True):
                st.session_state.combat_history = []
                st.success("Statistics cleared!")

        if resolve_button:
            # Resolve combat with multiple weapons
            resolver = CombatResolver()

            if len(weapon_configs) == 1:
                # Single weapon - use original display logic
                result = resolver.resolve_combat(weapon_configs[0])

                # Track history
                st.session_state.combat_history.append({
                    'actual': result.total_damage,
                    'expected': result.expected_damage
                })

                # Display results
                st.markdown("---")
                st.subheader("Combat Results")
                st.write(f"**Weapon: {result.weapon_name}**")

                # Phase results (single weapon display - existing logic follows below)
                multi_result = None
            else:
                # Multiple weapons - resolve and aggregate
                multi_result = resolver.resolve_multi_weapon_combat(weapon_configs)

                # Track history with combined result
                st.session_state.combat_history.append({
                    'actual': multi_result.total_damage,
                    'expected': multi_result.total_expected_damage
                })

                # Display aggregated results
                st.markdown("---")
                st.subheader("Combat Results - All Weapons")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Damage", multi_result.total_damage)
                with col2:
                    st.metric("Normal Damage", multi_result.total_normal_damage)
                with col3:
                    st.metric("Mortal Wounds", multi_result.total_mortal_wounds)
                with col4:
                    st.metric("Expected", f"{multi_result.total_expected_damage:.2f}")

                # Individual weapon results
                st.markdown("---")
                st.subheader("Per-Weapon Results")

                for idx, result in enumerate(multi_result.weapon_results):
                    weapon_config = weapon_configs[idx]
                    with st.expander(f"⚔️ {result.weapon_name}", expanded=False):
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Damage", result.total_damage)
                        with col2:
                            st.metric("Normal", result.normal_damage)
                        with col3:
                            st.metric("Mortals", result.mortal_wounds)
                        with col4:
                            st.metric("Expected", f"{result.expected_damage:.2f}")

                        st.markdown("---")

                        # Phase breakdowns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**HITS ({weapon_config.hit_target}+)**")
                            st.write(f"Rolled: {len(result.hit_roll.dice)}")
                            st.write(f"✓ Successes: {result.hit_roll.num_successes}")
                            st.write(f"✗ Failures: {result.hit_roll.num_failures}")

                        with col2:
                            st.write(f"**WOUNDS ({weapon_config.wound_target}+)**")
                            st.write(f"Rolled: {len(result.wound_roll.dice)}")
                            st.write(f"✓ Successes: {result.wound_roll.num_successes}")
                            st.write(f"✗ Failures: {result.wound_roll.num_failures}")

                        with col3:
                            st.write(f"**SAVES ({weapon_config.save_target}+)**")
                            st.write(f"Rolled: {len(result.save_roll.dice)}")
                            st.write(f"✓ Saved: {result.save_roll.num_successes}")
                            st.write(f"✗ Failed: {result.save_roll.num_failures}")

                        # Damage rolls if variable
                        if result.normal_damage_rolls and isinstance(weapon_config.damage, str):
                            st.markdown("---")
                            st.write(f"**Normal Damage Rolls:** {result.normal_damage_rolls}")
                            st.write(f"Total normal damage points: {sum(result.normal_damage_rolls)}")

                        # Ward saves if present
                        if result.ward_roll is not None:
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**WARD SAVES ({weapon_config.ward_target}+)**")
                                st.write(f"Damage points: {len(result.ward_roll.dice)}")
                                st.write(f"✓ Warded: {result.ward_roll.num_successes}")
                                st.write(f"✗ Damage taken: {result.ward_roll.num_failures}")
                            with col2:
                                if result.normal_damage_rolls and isinstance(weapon_config.damage, str):
                                    st.write(f"**Damage rolled:** {result.normal_damage_rolls}")

                        # Mortal wounds breakdown
                        if result.mortal_wounds > 0:
                            st.markdown("---")
                            st.write(f"**MORTAL WOUNDS: {result.mortal_wounds}**")
                            if result.mortal_damage_rolls and isinstance(weapon_config.damage, str):
                                st.write(f"Mortal damage rolls: {result.mortal_damage_rolls}")
                                st.write(f"Total mortal damage points: {sum(result.mortal_damage_rolls)}")
                            if result.mortal_ward_roll:
                                st.write(f"Ward saves vs mortals - Damage points: {len(result.mortal_ward_roll.dice)}, Warded: {result.mortal_ward_roll.num_successes}, Failed: {result.mortal_ward_roll.num_failures}")

                # Statistical Analysis for multi-weapon
                st.markdown("---")
                st.subheader("📊 Statistical Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**This Combat:**")
                    st.metric("Expected Damage", f"{multi_result.total_expected_damage:.2f}")
                    st.metric("Actual Damage", multi_result.total_damage)
                    deviation = multi_result.total_damage - multi_result.total_expected_damage
                    deviation_pct = (deviation / multi_result.total_expected_damage * 100) if multi_result.total_expected_damage > 0 else 0
                    st.metric("Deviation", f"{deviation:+.2f} ({deviation_pct:+.1f}%)")

                    if deviation > 0:
                        st.success("Above expected (lucky rolls!) ↑")
                    elif deviation < 0:
                        st.warning("Below expected (unlucky rolls) ↓")
                    else:
                        st.info("Exactly as expected!")

                with col2:
                    if len(st.session_state.combat_history) > 1:
                        st.write("**Cumulative Statistics:**")
                        actual_damages = [h['actual'] for h in st.session_state.combat_history]
                        expected_damages = [h['expected'] for h in st.session_state.combat_history]

                        total_actual = sum(actual_damages)
                        total_expected = sum(expected_damages)
                        total_deviation = total_actual - total_expected

                        st.metric("Combats Resolved", len(st.session_state.combat_history))
                        st.metric("Total Actual Damage", total_actual)
                        st.metric("Total Expected Damage", f"{total_expected:.2f}")
                        st.metric("Total Deviation", f"{total_deviation:+.2f}")

                        convergence_pct = abs(total_deviation / total_expected * 100) if total_expected > 0 else 0
                        if convergence_pct < 5:
                            st.success(f"Convergence: Excellent ({convergence_pct:.1f}% off) ✓")
                        elif convergence_pct < 10:
                            st.info(f"Convergence: Good ({convergence_pct:.1f}% off)")
                        elif convergence_pct < 20:
                            st.warning(f"Convergence: Moderate ({convergence_pct:.1f}% off)")
                        else:
                            st.error(f"Need more data ({convergence_pct:.1f}% off)")

                # Convergence visualization
                if len(st.session_state.combat_history) > 1:
                    st.markdown("---")
                    st.subheader("📈 Convergence Visualization")
                    fig = plot_convergence(st.session_state.combat_history)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

            # Detailed phase results (only for single weapon)
            if multi_result is None:
                config = weapon_configs[0]
                with st.expander(f"PHASE 1: TO HIT ({config.hit_target}+)", expanded=True):
                    if config.hit_reroll_ones:
                        st.write("- Reroll 1s enabled")
                    if config.hit_reroll_fails:
                        st.write("- Reroll all fails enabled")
                    if config.hit_crits_2_hits:
                        st.write(f"- {config.hit_crit_value}+ count as 2 hits (crits)")
                    if config.hit_auto_wound_on_crit:
                        st.write(f"- {config.hit_crit_value}+ auto-wound")
                    if config.hit_mortal_wounds_on_crit:
                        st.write(f"- {config.hit_crit_value}+ = mortal wounds")
                        if config.hit_mortals_continue:
                            st.write("  → Mortals + continue to normal damage")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rolled", len(result.hit_roll.dice))
                    with col2:
                        st.metric("Successes", result.hit_roll.num_successes)
                    with col3:
                        st.metric("Failures", result.hit_roll.num_failures)

                    if len(result.hit_roll.dice) <= 30:
                        st.code(str(result.hit_roll.dice.tolist()))

                with st.expander(f"PHASE 2: TO WOUND ({config.wound_target}+)"):
                    if config.wound_reroll_ones:
                        st.write("- Reroll 1s enabled")
                    if config.wound_reroll_fails:
                        st.write("- Reroll all fails enabled")
                    if config.wound_mortal_wounds_on_crit:
                        st.write(f"- {config.wound_crit_value}+ = mortal wounds")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rolled", len(result.wound_roll.dice))
                    with col2:
                        st.metric("Successes", result.wound_roll.num_successes)
                    with col3:
                        st.metric("Failures", result.wound_roll.num_failures)

                    if len(result.wound_roll.dice) <= 30 and len(result.wound_roll.dice) > 0:
                        st.code(str(result.wound_roll.dice.tolist()))

                with st.expander(f"PHASE 3: SAVES ({config.save_target}+) [Defender]"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rolled", len(result.save_roll.dice))
                    with col2:
                        st.metric("Successful Saves", result.save_roll.num_successes)
                    with col3:
                        st.metric("Failed Saves", result.save_roll.num_failures)

                    if len(result.save_roll.dice) <= 30 and len(result.save_roll.dice) > 0:
                        st.code(str(result.save_roll.dice.tolist()))

                    # Show damage rolls if variable damage
                    if result.normal_damage_rolls and isinstance(config.damage, str):
                        st.write(f"**Damage rolls per failed save:** {result.normal_damage_rolls}")
                        st.write(f"Total damage points: {sum(result.normal_damage_rolls)}")

                if result.ward_roll is not None:
                    with st.expander(f"PHASE 4: WARD SAVES ({config.ward_target}+) [Defender - vs Damage Points]"):
                        st.write("Ward saves are rolled against individual damage points (after damage is rolled)")

                        # Show how damage was generated if variable
                        if result.normal_damage_rolls and isinstance(config.damage, str):
                            st.write(f"**Damage rolled:** {result.normal_damage_rolls} → Total: {sum(result.normal_damage_rolls)} points")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Damage Points to Ward", len(result.ward_roll.dice))
                        with col2:
                            st.metric("Successful Wards", result.ward_roll.num_successes)
                        with col3:
                            st.metric("Damage Taken", result.ward_roll.num_failures)

                        if len(result.ward_roll.dice) <= 30:
                            st.code(str(result.ward_roll.dice.tolist()))

                if result.mortal_wounds > 0:
                    with st.expander("MORTAL WOUNDS"):
                        # Show damage rolls if variable damage
                        if result.mortal_damage_rolls and isinstance(config.damage, str):
                            st.write(f"**Damage rolls per mortal wound:** {result.mortal_damage_rolls}")
                            st.write(f"Total mortal damage points: {sum(result.mortal_damage_rolls)}")

                        if result.mortal_ward_roll:
                            st.write(f"Mortal damage points rolled, then ward saves against each point")
                            st.write(f"Damage points: {len(result.mortal_ward_roll.dice)}, Warded: {result.mortal_ward_roll.num_successes}, Failed: {result.mortal_ward_roll.num_failures}")
                        st.metric("Final Mortal Damage", result.mortal_wounds)

                # Final damage
                st.markdown("---")
                st.subheader("FINAL DAMAGE")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if result.mortal_wounds > 0:
                        st.metric("Normal Damage", result.normal_damage)
                        st.metric("Mortal Wounds", result.mortal_wounds)
                with col2:
                    st.metric("TOTAL DAMAGE", result.total_damage, delta=None)
                with col3:
                    expected_attacks = expected_value_from_dice(config.num_attacks)
                    expected_dmg_per_hit = expected_value_from_dice(config.damage) + config.damage_modifier
                    total_potential = expected_attacks * expected_dmg_per_hit
                    efficiency = (result.total_damage / total_potential * 100) if total_potential > 0 else 0
                    st.metric("Efficiency", f"{efficiency:.1f}%")

                # Statistical Analysis for single weapon
                st.markdown("---")
                st.subheader("📊 Statistical Analysis")

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**This Combat:**")
                    st.metric("Expected Damage", f"{result.expected_damage:.2f}")
                    st.metric("Actual Damage", result.total_damage)
                    deviation = result.total_damage - result.expected_damage
                    deviation_pct = (deviation / result.expected_damage * 100) if result.expected_damage > 0 else 0
                    st.metric("Deviation", f"{deviation:+.2f} ({deviation_pct:+.1f}%)")

                    if deviation > 0:
                        st.success("Above expected (lucky rolls!) ↑")
                    elif deviation < 0:
                        st.warning("Below expected (unlucky rolls) ↓")
                    else:
                        st.info("Exactly as expected!")

                with col2:
                    if len(st.session_state.combat_history) > 1:
                        st.write("**Cumulative Statistics:**")
                        actual_damages = [h['actual'] for h in st.session_state.combat_history]
                        expected_damages = [h['expected'] for h in st.session_state.combat_history]

                        total_actual = sum(actual_damages)
                        total_expected = sum(expected_damages)
                        total_deviation = total_actual - total_expected

                        st.metric("Combats Resolved", len(st.session_state.combat_history))
                        st.metric("Total Actual Damage", total_actual)
                        st.metric("Total Expected Damage", f"{total_expected:.2f}")
                        st.metric("Total Deviation", f"{total_deviation:+.2f}")

                        convergence_pct = abs(total_deviation / total_expected * 100) if total_expected > 0 else 0
                        if convergence_pct < 5:
                            st.success(f"Convergence: Excellent ({convergence_pct:.1f}% off) ✓")
                        elif convergence_pct < 10:
                            st.info(f"Convergence: Good ({convergence_pct:.1f}% off)")
                        elif convergence_pct < 20:
                            st.warning(f"Convergence: Moderate ({convergence_pct:.1f}% off)")
                        else:
                            st.error(f"Need more data ({convergence_pct:.1f}% off)")

                # Convergence visualization
                if len(st.session_state.combat_history) > 1:
                    st.markdown("---")
                    st.subheader("📈 Convergence Visualization")
                    fig = plot_convergence(st.session_state.combat_history)
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up to avoid memory issues


if __name__ == "__main__":
    main()
