import tkinter as tk
from tkinter import ttk
import numpy as np
from collections import Counter
from statistics import median, mode, StatisticsError
from dataclasses import dataclass
from typing import List, Optional

# Constants
MIN_DICE = 1
MAX_DICE = 1000
DICE_MIN = 1
DICE_MAX = 6
TEXT_HEIGHT = 20
TEXT_WIDTH = 70
PADDING = 10


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


class DiceRoller:
    """Handles dice rolling logic and statistics"""

    def __init__(self):
        self.rng = np.random.default_rng()
        self.roll_history = []

    def roll(self, num_dice):
        """Roll the specified number of d6 dice"""
        if num_dice < MIN_DICE or num_dice > MAX_DICE:
            raise ValueError(f"Number of dice must be between {MIN_DICE} and {MAX_DICE}")

        result = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=num_dice)
        self.roll_history.append(result)
        return result

    def roll_with_target(self, num_dice, target, modifier=0, reroll_ones=False,
                         reroll_all_fails=False, reroll_specific=None):
        """
        Roll dice and check against target with modifiers and rerolls

        Args:
            num_dice: Number of dice to roll
            target: Target number to meet or exceed (e.g., 3 for 3+)
            modifier: Modifier to add to each die roll (+1, -1, etc.)
            reroll_ones: If True, reroll all 1s once
            reroll_all_fails: If True, reroll all failures once
            reroll_specific: List of specific values to reroll (e.g., [1, 2])
        """
        if num_dice < MIN_DICE or num_dice > MAX_DICE:
            raise ValueError(f"Number of dice must be between {MIN_DICE} and {MAX_DICE}")

        if num_dice == 0:
            return RollResult(
                dice=np.array([]),
                successes=np.array([]),
                failures=np.array([]),
                num_successes=0,
                num_failures=0,
                target=target,
                modifier=modifier,
                rerolled_indices=[]
            )

        # Initial roll
        dice = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=num_dice)
        rerolled_indices = []

        # Handle rerolls
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

        # Apply modifier
        modified_dice = dice + modifier

        # Check against target (before reroll_all_fails to determine what to reroll)
        if reroll_all_fails and not reroll_ones and not reroll_specific:
            fail_mask = modified_dice < target
            reroll_indices = np.where(fail_mask)[0]
            dice[fail_mask] = self.rng.integers(DICE_MIN, DICE_MAX + 1, size=np.sum(fail_mask))
            modified_dice = dice + modifier
            rerolled_indices.extend(reroll_indices.tolist())

        # Final success/failure determination
        success_mask = modified_dice >= target
        successes = dice[success_mask]
        failures = dice[~success_mask]

        return RollResult(
            dice=dice,
            successes=successes,
            failures=failures,
            num_successes=np.sum(success_mask),
            num_failures=np.sum(~success_mask),
            target=target,
            modifier=modifier,
            rerolled_indices=list(set(rerolled_indices))
        )

    def get_statistics(self, dice_result):
        """Calculate statistics for a dice roll"""
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

    def get_historic_mean(self):
        """Calculate mean across all historic rolls"""
        if not self.roll_history:
            return None

        all_rolls = np.concatenate(self.roll_history)
        return round(np.mean(all_rolls), 3)

    def clear_history(self):
        """Clear roll history"""
        self.roll_history = []


@dataclass
class CombatConfig:
    """Configuration for combat resolution"""
    num_attacks: int
    hit_target: int = 4
    hit_modifier: int = 0
    hit_reroll_ones: bool = False
    hit_reroll_fails: bool = False
    hit_auto_wound_on_6: bool = False  # 6s to hit auto-wound
    hit_mortal_wounds_on_6: bool = False  # 6s to hit deal mortal wounds
    wound_target: int = 4
    wound_modifier: int = 0
    wound_reroll_ones: bool = False
    wound_reroll_fails: bool = False
    save_target: int = 4
    save_modifier: int = 0
    ward_target: Optional[int] = None
    ward_modifier: int = 0
    damage: int = 1  # Base damage per attack
    damage_modifier: int = 0  # Additional damage (e.g., +1 damage)


@dataclass
class CombatResult:
    """Result of full combat resolution"""
    hit_roll: RollResult
    wound_roll: RollResult
    save_roll: RollResult
    ward_roll: Optional[RollResult]
    normal_damage: int
    mortal_wounds: int
    mortal_ward_roll: Optional[RollResult]  # Ward saves against mortals
    total_damage: int
    expected_damage: float  # Theoretical expected damage


class CombatResolver:
    """Handles multi-stage combat resolution"""

    def __init__(self):
        self.dice_roller = DiceRoller()
        self.combat_history = []  # Track all combat resolutions

    def calculate_expected_damage(self, config: CombatConfig) -> float:
        """
        Calculate theoretical expected damage based on probabilities

        Args:
            config: Combat configuration

        Returns:
            Expected damage value
        """
        damage_per_hit = config.damage + config.damage_modifier

        # Calculate base hit probability
        base_hit_prob = max(0, min(1, (7 - config.hit_target + config.hit_modifier) / 6))

        # Adjust for rerolls
        if config.hit_reroll_ones:
            # Reroll 1s: if you roll a 1, you get another chance
            prob_roll_one = 1/6
            hit_prob = base_hit_prob + (prob_roll_one * base_hit_prob)
        elif config.hit_reroll_fails:
            # Reroll all fails: if you fail, you get another chance
            hit_prob = base_hit_prob + ((1 - base_hit_prob) * base_hit_prob)
        else:
            hit_prob = base_hit_prob

        # Handle special 6s
        prob_natural_6 = 1/6
        mortal_wound_damage = 0
        normal_hit_prob = hit_prob

        if config.hit_mortal_wounds_on_6:
            # 6s deal mortal wounds (skip wound and save)
            # These go straight to ward (or damage)
            prob_6_hits = prob_natural_6  # Natural 6 always hits
            normal_hit_prob = hit_prob - prob_6_hits  # Remove 6s from normal flow

            # Mortal wounds face only ward saves
            if config.ward_target:
                effective_ward_target = config.ward_target - config.ward_modifier
                ward_success_prob_mw = max(0, min(1, (7 - effective_ward_target) / 6))
                ward_fail_prob_mw = 1 - ward_success_prob_mw
                mortal_wound_damage = config.num_attacks * prob_6_hits * ward_fail_prob_mw * damage_per_hit
            else:
                mortal_wound_damage = config.num_attacks * prob_6_hits * damage_per_hit

        elif config.hit_auto_wound_on_6:
            # 6s auto-wound (skip wound roll, go to saves)
            prob_6_hits = prob_natural_6
            auto_wound_damage_contribution = prob_6_hits
            normal_hit_prob = hit_prob - prob_6_hits
        else:
            auto_wound_damage_contribution = 0

        # Calculate wound probability (for non-special hits)
        base_wound_prob = max(0, min(1, (7 - config.wound_target + config.wound_modifier) / 6))

        if config.wound_reroll_ones:
            prob_roll_one = 1/6
            wound_prob = base_wound_prob + (prob_roll_one * base_wound_prob)
        elif config.wound_reroll_fails:
            wound_prob = base_wound_prob + ((1 - base_wound_prob) * base_wound_prob)
        else:
            wound_prob = base_wound_prob

        # Calculate save probability (defender's perspective)
        # Save succeeds if roll >= target (modified by save modifier)
        # Negative save modifier makes it harder to save (increases effective target)
        effective_save_target = config.save_target - config.save_modifier
        save_success_prob = max(0, min(1, (7 - effective_save_target) / 6))
        save_fail_prob = 1 - save_success_prob

        # Calculate ward probability (if applicable)
        if config.ward_target:
            effective_ward_target = config.ward_target - config.ward_modifier
            ward_success_prob = max(0, min(1, (7 - effective_ward_target) / 6))
            ward_fail_prob = 1 - ward_success_prob
        else:
            ward_fail_prob = 1  # No ward = all get through

        # Expected normal damage
        if config.hit_auto_wound_on_6:
            # Normal hits go through wound phase
            normal_damage_contribution = (normal_hit_prob * wound_prob)
            # Auto-wounds skip wound phase
            auto_wound_contribution = auto_wound_damage_contribution
            # Both face saves and wards
            expected_normal_damage = config.num_attacks * (normal_damage_contribution + auto_wound_contribution) * save_fail_prob * ward_fail_prob * damage_per_hit
        else:
            expected_normal_damage = config.num_attacks * normal_hit_prob * wound_prob * save_fail_prob * ward_fail_prob * damage_per_hit

        # Total expected damage
        expected_total = expected_normal_damage + mortal_wound_damage

        return round(expected_total, 2)

    def resolve_combat(self, config: CombatConfig) -> CombatResult:
        """
        Resolve full combat sequence: Hit -> Wound -> Save -> Ward
        Handles special rules like auto-wounds and mortal wounds on 6s

        Args:
            config: Combat configuration with all parameters

        Returns:
            CombatResult with all stage results
        """
        # Calculate actual damage per hit
        damage_per_hit = config.damage + config.damage_modifier

        # Stage 1: Hit rolls
        hit_roll = self.dice_roller.roll_with_target(
            config.num_attacks,
            config.hit_target,
            config.hit_modifier,
            config.hit_reroll_ones,
            config.hit_reroll_fails
        )

        # Check for special 6s rules
        natural_6s_that_hit = 0
        mortal_wound_count = 0
        auto_wound_count = 0
        normal_hits = hit_roll.num_successes

        if (config.hit_auto_wound_on_6 or config.hit_mortal_wounds_on_6) and len(hit_roll.successes) > 0:
            # Count natural 6s among successful hits
            natural_6s_that_hit = np.sum(hit_roll.successes == 6)

            if config.hit_mortal_wounds_on_6:
                # Mortal wounds: skip wound and save, just apply damage
                mortal_wound_count = natural_6s_that_hit
                normal_hits -= natural_6s_that_hit  # Remove from normal processing
            elif config.hit_auto_wound_on_6:
                # Auto-wounds: skip wound roll, go straight to saves
                auto_wound_count = natural_6s_that_hit
                normal_hits -= natural_6s_that_hit

        # Stage 2: Wound rolls (for non-special hits)
        wound_roll = self.dice_roller.roll_with_target(
            normal_hits,
            config.wound_target,
            config.wound_modifier,
            config.wound_reroll_ones,
            config.wound_reroll_fails
        )

        # Add auto-wounds to wound successes
        total_wounds = wound_roll.num_successes + auto_wound_count

        # Stage 3: Save rolls (defender rolls against successful wounds)
        save_roll = self.dice_roller.roll_with_target(
            total_wounds,
            config.save_target,
            config.save_modifier,
            reroll_ones=False,
            reroll_all_fails=False
        )

        # Failed saves become potential damage
        failed_saves = save_roll.num_failures

        # Stage 4: Ward saves against normal damage
        ward_roll = None
        normal_damage = 0

        if config.ward_target is not None and failed_saves > 0:
            ward_roll = self.dice_roller.roll_with_target(
                failed_saves,
                config.ward_target,
                config.ward_modifier,
                reroll_ones=False,
                reroll_all_fails=False
            )
            # Failed ward saves become damage
            normal_damage = ward_roll.num_failures * damage_per_hit
        else:
            normal_damage = failed_saves * damage_per_hit

        # Stage 5: Ward saves against mortal wounds (if applicable)
        mortal_ward_roll = None
        final_mortal_wounds = mortal_wound_count * damage_per_hit

        if config.ward_target is not None and mortal_wound_count > 0:
            mortal_ward_roll = self.dice_roller.roll_with_target(
                mortal_wound_count,
                config.ward_target,
                config.ward_modifier,
                reroll_ones=False,
                reroll_all_fails=False
            )
            # Failed ward saves become mortal wounds
            final_mortal_wounds = mortal_ward_roll.num_failures * damage_per_hit

        total_damage = normal_damage + final_mortal_wounds

        # Calculate expected damage
        expected_damage = self.calculate_expected_damage(config)

        result = CombatResult(
            hit_roll=hit_roll,
            wound_roll=wound_roll,
            save_roll=save_roll,
            ward_roll=ward_roll,
            normal_damage=normal_damage,
            mortal_wounds=final_mortal_wounds,
            mortal_ward_roll=mortal_ward_roll,
            total_damage=total_damage,
            expected_damage=expected_damage
        )

        # Track combat history
        self.combat_history.append({
            'config': config,
            'actual_damage': total_damage,
            'expected_damage': expected_damage
        })

        return result

    def get_statistics(self):
        """Get statistics from combat history"""
        if not self.combat_history:
            return None

        actual_damages = [h['actual_damage'] for h in self.combat_history]
        expected_damages = [h['expected_damage'] for h in self.combat_history]

        return {
            'num_combats': len(self.combat_history),
            'mean_actual': np.mean(actual_damages),
            'mean_expected': np.mean(expected_damages),
            'std_actual': np.std(actual_damages) if len(actual_damages) > 1 else 0,
            'total_actual': sum(actual_damages),
            'total_expected': sum(expected_damages),
            'deviation': np.mean(actual_damages) - np.mean(expected_damages)
        }

    def clear_history(self):
        """Clear combat history"""
        self.combat_history = []


class DiceRollerApp:
    """GUI application for rolling dice"""

    def __init__(self, master):
        self.master = master
        master.title("D6 Dice Roller & Combat Resolver")
        master.minsize(700, 600)

        self.dice_roller = DiceRoller()
        self.combat_resolver = CombatResolver()

        # Font size management
        self.font_size = tk.IntVar(value=9)
        self.font_family = 'Courier'

        self._setup_ui()

    def _setup_ui(self):
        """Setup the user interface with tabs"""
        # Configure main window grid
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(1, weight=1)

        # Font size controls at top
        font_frame = ttk.Frame(self.master, padding=5)
        font_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))

        ttk.Label(font_frame, text="Text Size:").pack(side=tk.LEFT, padx=5)

        ttk.Button(font_frame, text="-", width=3, command=self._decrease_font_size).pack(side=tk.LEFT, padx=2)

        font_size_label = ttk.Label(font_frame, textvariable=self.font_size, width=3)
        font_size_label.pack(side=tk.LEFT, padx=2)

        ttk.Button(font_frame, text="+", width=3, command=self._increase_font_size).pack(side=tk.LEFT, padx=2)

        ttk.Button(font_frame, text="Reset", command=self._reset_font_size).pack(side=tk.LEFT, padx=10)

        # Create notebook for tabs
        notebook = ttk.Notebook(self.master)
        notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Create tabs
        simple_tab = ttk.Frame(notebook, padding=PADDING)
        combat_tab = ttk.Frame(notebook, padding=PADDING)

        notebook.add(simple_tab, text="Simple Rolls")
        notebook.add(combat_tab, text="Combat Resolver")

        # Setup simple roll tab
        self._setup_simple_tab(simple_tab)

        # Setup combat tab
        self._setup_combat_tab(combat_tab)

    def _setup_simple_tab(self, parent):
        """Setup the simple dice rolling tab"""
        # Configure grid weights
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        # Input section
        input_label = ttk.Label(parent, text="Number of d6s:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=5)

        self.dice_count_entry = ttk.Entry(parent, width=15)
        self.dice_count_entry.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        self.dice_count_entry.bind('<Return>', lambda e: self.roll_dice())

        # Button frame
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        self.roll_button = ttk.Button(button_frame, text="Roll Dice", command=self.roll_dice)
        self.roll_button.grid(row=0, column=0, padx=5)

        self.clear_button = ttk.Button(button_frame, text="Clear History", command=self.clear_history)
        self.clear_button.grid(row=0, column=1, padx=5)

        # Quick roll buttons
        quick_frame = ttk.LabelFrame(button_frame, text="Quick Roll", padding=5)
        quick_frame.grid(row=0, column=2, padx=10)

        for i, num in enumerate([5, 10, 20]):
            btn = ttk.Button(quick_frame, text=f"{num}d6", width=6,
                           command=lambda n=num: self._quick_roll(n))
            btn.grid(row=0, column=i, padx=2)

        # Results display
        results_frame = ttk.LabelFrame(parent, text="Results", padding=5)
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_text = tk.Text(results_frame, height=TEXT_HEIGHT, width=TEXT_WIDTH,
                                    wrap=tk.WORD, font=(self.font_family, self.font_size.get()))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.dice_count_entry.focus()

    def _setup_combat_tab(self, parent):
        """Setup the combat resolution tab"""
        # Configure grid
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # Config frame
        config_frame = ttk.LabelFrame(parent, text="Combat Configuration", padding=10)
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Attacks and Damage
        row = 0
        ttk.Label(config_frame, text="Number of Attacks:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.combat_attacks = ttk.Entry(config_frame, width=10)
        self.combat_attacks.grid(row=row, column=1, sticky=tk.W, padx=5)
        self.combat_attacks.insert(0, "10")

        ttk.Label(config_frame, text="Damage:").grid(row=row, column=3, sticky=tk.W, padx=(10, 0))
        self.damage = ttk.Combobox(config_frame, values=[1, 2, 3, 4, 5, 6], width=5, state='readonly')
        self.damage.set(1)
        self.damage.grid(row=row, column=4, sticky=tk.W, padx=5)

        ttk.Label(config_frame, text="Damage Modifier:").grid(row=row, column=5, sticky=tk.W, padx=(10, 0))
        self.damage_modifier = ttk.Combobox(config_frame, values=[-2, -1, 0, 1, 2, 3], width=5, state='readonly')
        self.damage_modifier.set(0)
        self.damage_modifier.grid(row=row, column=6, sticky=tk.W, padx=5)

        # Hit section
        row += 1
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)

        row += 1
        ttk.Label(config_frame, text="TO HIT", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=6, pady=3)

        row += 1
        ttk.Label(config_frame, text="Hit on:").grid(row=row, column=0, sticky=tk.W)
        self.hit_target = ttk.Combobox(config_frame, values=[2, 3, 4, 5, 6], width=5, state='readonly')
        self.hit_target.set(3)
        self.hit_target.grid(row=row, column=1, sticky=tk.W, padx=5)
        ttk.Label(config_frame, text="+").grid(row=row, column=2)

        ttk.Label(config_frame, text="Modifier:").grid(row=row, column=3, sticky=tk.W, padx=(10, 0))
        self.hit_modifier = ttk.Combobox(config_frame, values=[-2, -1, 0, 1, 2], width=5, state='readonly')
        self.hit_modifier.set(0)
        self.hit_modifier.grid(row=row, column=4, sticky=tk.W, padx=5)

        row += 1
        self.hit_reroll_ones = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="Reroll 1s", variable=self.hit_reroll_ones).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.hit_reroll_fails = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="Reroll all fails", variable=self.hit_reroll_fails).grid(row=row, column=2, columnspan=3, sticky=tk.W, pady=2)

        row += 1
        self.hit_auto_wound_on_6 = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="6s auto-wound (skip wound roll)",
                       variable=self.hit_auto_wound_on_6,
                       command=self._toggle_special_6s).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=2)

        self.hit_mortal_wounds_on_6 = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="6s = mortal wounds (skip wound+save)",
                       variable=self.hit_mortal_wounds_on_6,
                       command=self._toggle_special_6s).grid(row=row, column=3, columnspan=3, sticky=tk.W, pady=2)

        # Wound section
        row += 1
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)

        row += 1
        ttk.Label(config_frame, text="TO WOUND", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=6, pady=3)

        row += 1
        ttk.Label(config_frame, text="Wound on:").grid(row=row, column=0, sticky=tk.W)
        self.wound_target = ttk.Combobox(config_frame, values=[2, 3, 4, 5, 6], width=5, state='readonly')
        self.wound_target.set(4)
        self.wound_target.grid(row=row, column=1, sticky=tk.W, padx=5)
        ttk.Label(config_frame, text="+").grid(row=row, column=2)

        ttk.Label(config_frame, text="Modifier:").grid(row=row, column=3, sticky=tk.W, padx=(10, 0))
        self.wound_modifier = ttk.Combobox(config_frame, values=[-2, -1, 0, 1, 2], width=5, state='readonly')
        self.wound_modifier.set(0)
        self.wound_modifier.grid(row=row, column=4, sticky=tk.W, padx=5)

        row += 1
        self.wound_reroll_ones = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="Reroll 1s", variable=self.wound_reroll_ones).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.wound_reroll_fails = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="Reroll all fails", variable=self.wound_reroll_fails).grid(row=row, column=2, columnspan=3, sticky=tk.W, pady=2)

        # Save section
        row += 1
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)

        row += 1
        ttk.Label(config_frame, text="SAVES (Defender)", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=6, pady=3)

        row += 1
        ttk.Label(config_frame, text="Save on:").grid(row=row, column=0, sticky=tk.W)
        self.save_target = ttk.Combobox(config_frame, values=[2, 3, 4, 5, 6], width=5, state='readonly')
        self.save_target.set(4)
        self.save_target.grid(row=row, column=1, sticky=tk.W, padx=5)
        ttk.Label(config_frame, text="+").grid(row=row, column=2)

        ttk.Label(config_frame, text="Save Modifier:").grid(row=row, column=3, sticky=tk.W, padx=(10, 0))
        self.save_modifier = ttk.Combobox(config_frame, values=[-2, -1, 0, 1, 2], width=5, state='readonly')
        self.save_modifier.set(0)
        self.save_modifier.grid(row=row, column=4, sticky=tk.W, padx=5)

        # Ward section
        row += 1
        ttk.Separator(config_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=5)

        row += 1
        ttk.Label(config_frame, text="WARD SAVE (Optional)", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=6, pady=3)

        row += 1
        self.ward_enabled = tk.BooleanVar()
        ttk.Checkbutton(config_frame, text="Enable Ward", variable=self.ward_enabled,
                       command=self._toggle_ward).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=2)

        ttk.Label(config_frame, text="Ward on:").grid(row=row, column=2, sticky=tk.W)
        self.ward_target = ttk.Combobox(config_frame, values=[2, 3, 4, 5, 6], width=5, state='readonly')
        self.ward_target.set(5)
        self.ward_target.grid(row=row, column=3, sticky=tk.W, padx=5)
        self.ward_target.config(state='disabled')

        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, pady=10)

        ttk.Button(button_frame, text="Resolve Combat", command=self.resolve_combat).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_combat_results).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Clear Statistics", command=self.clear_combat_statistics).grid(row=0, column=2, padx=5)

        # Combat results display
        results_frame = ttk.LabelFrame(parent, text="Combat Results", padding=5)
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.combat_results_text = tk.Text(results_frame, height=TEXT_HEIGHT, width=TEXT_WIDTH,
                                           wrap=tk.WORD, font=(self.font_family, self.font_size.get()))
        self.combat_results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        combat_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.combat_results_text.yview)
        combat_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.combat_results_text.configure(yscrollcommand=combat_scrollbar.set)

    def _toggle_ward(self):
        """Enable/disable ward save controls"""
        if self.ward_enabled.get():
            self.ward_target.config(state='readonly')
        else:
            self.ward_target.config(state='disabled')

    def _toggle_special_6s(self):
        """Ensure auto-wound and mortal wounds on 6s are mutually exclusive"""
        if self.hit_auto_wound_on_6.get():
            self.hit_mortal_wounds_on_6.set(False)
        elif self.hit_mortal_wounds_on_6.get():
            self.hit_auto_wound_on_6.set(False)

    def _quick_roll(self, num):
        """Quick roll with predefined number"""
        self.dice_count_entry.delete(0, tk.END)
        self.dice_count_entry.insert(0, str(num))
        self.roll_dice()

    def roll_dice(self):
        """Handle dice roll button click"""
        try:
            num_dice = int(self.dice_count_entry.get())

            # Roll the dice
            dice_result = self.dice_roller.roll(num_dice)

            # Get statistics and distribution
            stats = self.dice_roller.get_statistics(dice_result)
            distribution = self.dice_roller.get_distribution(dice_result)

            # Display results
            self._display_results(num_dice, dice_result, stats, distribution)

        except ValueError as e:
            self._display_error(str(e))

    def _display_results(self, num_dice, dice_result, stats, distribution):
        """Display roll results in the text widget"""
        self.results_text.delete(1.0, tk.END)

        # Header
        self.results_text.insert(tk.END, f"{'=' * 50}\n")
        self.results_text.insert(tk.END, f"Rolling {num_dice} d6 dice\n")
        self.results_text.insert(tk.END, f"{'=' * 50}\n\n")

        # Individual rolls (show if reasonable number)
        if num_dice <= 50:
            self.results_text.insert(tk.END, f"Individual rolls:\n")
            rolls_str = str(list(dice_result))
            self.results_text.insert(tk.END, f"{rolls_str}\n\n")

        # Distribution
        self.results_text.insert(tk.END, f"Distribution:\n")
        for value, count in distribution:
            bar = '█' * count
            percentage = (count / num_dice) * 100
            self.results_text.insert(tk.END, f"  {value}: {bar} ({count}) - {percentage:.1f}%\n")

        # Statistics
        self.results_text.insert(tk.END, f"\nStatistics:\n")
        self.results_text.insert(tk.END, f"  Sum:    {stats['sum']}\n")
        self.results_text.insert(tk.END, f"  Mean:   {stats['mean']}\n")
        self.results_text.insert(tk.END, f"  Median: {stats['median']}\n")

        if stats['mode'] is not None:
            self.results_text.insert(tk.END, f"  Mode:   {stats['mode']}\n")

        self.results_text.insert(tk.END, f"  Min:    {stats['min']}\n")
        self.results_text.insert(tk.END, f"  Max:    {stats['max']}\n")

        # Historic mean
        historic_mean = self.dice_roller.get_historic_mean()
        if historic_mean is not None:
            num_sessions = len(self.dice_roller.roll_history)
            self.results_text.insert(tk.END, f"\nHistoric mean: {historic_mean} ")
            self.results_text.insert(tk.END, f"(across {num_sessions} roll{'s' if num_sessions > 1 else ''})\n")

    def _display_error(self, message):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}\n")
        self.results_text.insert(tk.END, f"\nPlease enter a number between {MIN_DICE} and {MAX_DICE}.")

    def clear_history(self):
        """Clear roll history"""
        self.dice_roller.clear_history()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "History cleared.\n")

    def resolve_combat(self):
        """Resolve combat sequence"""
        try:
            # Get configuration
            config = CombatConfig(
                num_attacks=int(self.combat_attacks.get()),
                hit_target=int(self.hit_target.get()),
                hit_modifier=int(self.hit_modifier.get()),
                hit_reroll_ones=self.hit_reroll_ones.get(),
                hit_reroll_fails=self.hit_reroll_fails.get(),
                hit_auto_wound_on_6=self.hit_auto_wound_on_6.get(),
                hit_mortal_wounds_on_6=self.hit_mortal_wounds_on_6.get(),
                wound_target=int(self.wound_target.get()),
                wound_modifier=int(self.wound_modifier.get()),
                wound_reroll_ones=self.wound_reroll_ones.get(),
                wound_reroll_fails=self.wound_reroll_fails.get(),
                save_target=int(self.save_target.get()),
                save_modifier=int(self.save_modifier.get()),
                ward_target=int(self.ward_target.get()) if self.ward_enabled.get() else None,
                ward_modifier=0,
                damage=int(self.damage.get()),
                damage_modifier=int(self.damage_modifier.get())
            )

            # Resolve combat
            result = self.combat_resolver.resolve_combat(config)

            # Display results
            self._display_combat_results(config, result)

        except ValueError as e:
            self.combat_results_text.delete(1.0, tk.END)
            self.combat_results_text.insert(tk.END, f"Error: {e}\n")

    def _display_combat_results(self, config: CombatConfig, result: CombatResult):
        """Display combat resolution results"""
        text = self.combat_results_text
        text.delete(1.0, tk.END)

        # Header
        text.insert(tk.END, f"{'=' * 60}\n")
        text.insert(tk.END, f"COMBAT RESOLUTION - {config.num_attacks} attacks\n")
        damage_str = f"Damage: {config.damage}"
        if config.damage_modifier != 0:
            damage_str += f" {config.damage_modifier:+d}"
        text.insert(tk.END, f"{damage_str}\n")
        text.insert(tk.END, f"{'=' * 60}\n\n")

        # Hit phase
        text.insert(tk.END, f"PHASE 1: TO HIT (Target: {config.hit_target}+")
        if config.hit_modifier != 0:
            text.insert(tk.END, f", Modifier: {config.hit_modifier:+d}")
        text.insert(tk.END, ")\n")

        if config.hit_reroll_ones:
            text.insert(tk.END, "  Reroll 1s enabled\n")
        if config.hit_reroll_fails:
            text.insert(tk.END, "  Reroll all fails enabled\n")
        if config.hit_auto_wound_on_6:
            text.insert(tk.END, "  6s to hit auto-wound (skip wound roll)\n")
        if config.hit_mortal_wounds_on_6:
            text.insert(tk.END, "  6s to hit deal mortal wounds (skip wound+save)\n")

        text.insert(tk.END, f"  Rolled {len(result.hit_roll.dice)} dice\n")
        if len(result.hit_roll.dice) <= 30:
            text.insert(tk.END, f"  Rolls: {list(result.hit_roll.dice)}\n")
        text.insert(tk.END, f"  Successes: {result.hit_roll.num_successes}\n")
        text.insert(tk.END, f"  Failures: {result.hit_roll.num_failures}\n")
        if result.hit_roll.rerolled_indices:
            text.insert(tk.END, f"  Rerolled: {len(result.hit_roll.rerolled_indices)} dice\n")
        text.insert(tk.END, "\n")

        # Wound phase
        text.insert(tk.END, f"PHASE 2: TO WOUND (Target: {config.wound_target}+")
        if config.wound_modifier != 0:
            text.insert(tk.END, f", Modifier: {config.wound_modifier:+d}")
        text.insert(tk.END, ")\n")

        if config.wound_reroll_ones:
            text.insert(tk.END, "  Reroll 1s enabled\n")
        if config.wound_reroll_fails:
            text.insert(tk.END, "  Reroll all fails enabled\n")

        text.insert(tk.END, f"  Rolled {len(result.wound_roll.dice)} dice (from successful hits)\n")
        if len(result.wound_roll.dice) <= 30 and len(result.wound_roll.dice) > 0:
            text.insert(tk.END, f"  Rolls: {list(result.wound_roll.dice)}\n")
        text.insert(tk.END, f"  Successes: {result.wound_roll.num_successes}\n")
        text.insert(tk.END, f"  Failures: {result.wound_roll.num_failures}\n")
        if result.wound_roll.rerolled_indices:
            text.insert(tk.END, f"  Rerolled: {len(result.wound_roll.rerolled_indices)} dice\n")
        text.insert(tk.END, "\n")

        # Save phase
        text.insert(tk.END, f"PHASE 3: SAVES (Target: {config.save_target}+")
        if config.save_modifier != 0:
            text.insert(tk.END, f", Modifier: {config.save_modifier:+d}")
        text.insert(tk.END, ") [Defender]\n")

        text.insert(tk.END, f"  Rolled {len(result.save_roll.dice)} dice (wounds to save)\n")
        if len(result.save_roll.dice) <= 30 and len(result.save_roll.dice) > 0:
            text.insert(tk.END, f"  Rolls: {list(result.save_roll.dice)}\n")
        text.insert(tk.END, f"  Successful Saves: {result.save_roll.num_successes}\n")
        text.insert(tk.END, f"  Failed Saves: {result.save_roll.num_failures}\n")
        text.insert(tk.END, "\n")

        # Ward phase (normal damage)
        if result.ward_roll is not None:
            text.insert(tk.END, f"PHASE 4: WARD SAVES vs Normal Damage (Target: {config.ward_target}+) [Defender]\n")
            text.insert(tk.END, f"  Rolled {len(result.ward_roll.dice)} dice (failed saves)\n")
            if len(result.ward_roll.dice) <= 30 and len(result.ward_roll.dice) > 0:
                text.insert(tk.END, f"  Rolls: {list(result.ward_roll.dice)}\n")
            text.insert(tk.END, f"  Successful Wards: {result.ward_roll.num_successes}\n")
            text.insert(tk.END, f"  Failed Wards: {result.ward_roll.num_failures}\n")
            text.insert(tk.END, "\n")

        # Mortal wounds phase
        if result.mortal_wounds > 0:
            text.insert(tk.END, f"MORTAL WOUNDS PHASE\n")
            if result.mortal_ward_roll is not None:
                text.insert(tk.END, f"  Ward saves vs Mortal Wounds (Target: {config.ward_target}+)\n")
                text.insert(tk.END, f"  Rolled {len(result.mortal_ward_roll.dice)} dice\n")
                if len(result.mortal_ward_roll.dice) <= 30:
                    text.insert(tk.END, f"  Rolls: {list(result.mortal_ward_roll.dice)}\n")
                text.insert(tk.END, f"  Successful Wards: {result.mortal_ward_roll.num_successes}\n")
                text.insert(tk.END, f"  Failed Wards: {result.mortal_ward_roll.num_failures}\n")
            text.insert(tk.END, f"  Final Mortal Wounds: {result.mortal_wounds}\n")
            text.insert(tk.END, "\n")

        # Final damage
        text.insert(tk.END, f"{'=' * 60}\n")
        text.insert(tk.END, f"TOTAL DAMAGE: {result.total_damage}\n")
        if result.mortal_wounds > 0:
            text.insert(tk.END, f"  Normal Damage:  {result.normal_damage}\n")
            text.insert(tk.END, f"  Mortal Wounds:  {result.mortal_wounds}\n")
        text.insert(tk.END, f"{'=' * 60}\n\n")

        # Summary table
        text.insert(tk.END, "SUMMARY:\n")
        text.insert(tk.END, f"  Attacks:        {config.num_attacks}\n")
        text.insert(tk.END, f"  Hits:           {result.hit_roll.num_successes}\n")
        text.insert(tk.END, f"  Wounds:         {result.wound_roll.num_successes}\n")
        text.insert(tk.END, f"  Failed Saves:   {result.save_roll.num_failures}\n")
        if result.ward_roll:
            text.insert(tk.END, f"  Failed Wards:   {result.ward_roll.num_failures}\n")
        if result.mortal_wounds > 0:
            text.insert(tk.END, f"  Normal Damage:  {result.normal_damage}\n")
            text.insert(tk.END, f"  Mortal Wounds:  {result.mortal_wounds}\n")
            text.insert(tk.END, f"  TOTAL DAMAGE:   {result.total_damage}\n")
        else:
            text.insert(tk.END, f"  Total Damage:   {result.total_damage}\n")

        # Efficiency calculation
        if config.num_attacks > 0:
            total_damage_potential = config.num_attacks * (config.damage + config.damage_modifier)
            efficiency = (result.total_damage / total_damage_potential) * 100
            text.insert(tk.END, f"\n  Damage Efficiency: {efficiency:.1f}%")
            text.insert(tk.END, f" ({result.total_damage}/{total_damage_potential} potential damage)\n")

        # Statistical analysis
        text.insert(tk.END, f"\n{'=' * 60}\n")
        text.insert(tk.END, f"STATISTICAL ANALYSIS\n")
        text.insert(tk.END, f"{'=' * 60}\n\n")

        text.insert(tk.END, f"This Combat:\n")
        text.insert(tk.END, f"  Expected Damage: {result.expected_damage:.2f}\n")
        text.insert(tk.END, f"  Actual Damage:   {result.total_damage}\n")
        deviation = result.total_damage - result.expected_damage
        deviation_pct = (deviation / result.expected_damage * 100) if result.expected_damage > 0 else 0
        text.insert(tk.END, f"  Deviation:       {deviation:+.2f} ({deviation_pct:+.1f}%)\n")

        if deviation > 0:
            text.insert(tk.END, f"  Result: Above expected (lucky rolls!) \u2191\n")
        elif deviation < 0:
            text.insert(tk.END, f"  Result: Below expected (unlucky rolls) \u2193\n")
        else:
            text.insert(tk.END, f"  Result: Exactly as expected!\n")

        # Historical statistics
        stats = self.combat_resolver.get_statistics()
        if stats and stats['num_combats'] > 1:
            text.insert(tk.END, f"\nCumulative Statistics ({stats['num_combats']} combats):\n")
            text.insert(tk.END, f"  Average Actual Damage:   {stats['mean_actual']:.2f}\n")
            text.insert(tk.END, f"  Average Expected Damage: {stats['mean_expected']:.2f}\n")
            text.insert(tk.END, f"  Overall Deviation:       {stats['deviation']:+.2f}\n")
            text.insert(tk.END, f"  Standard Deviation:      {stats['std_actual']:.2f}\n")

            text.insert(tk.END, f"\n  Total Actual:   {stats['total_actual']}\n")
            text.insert(tk.END, f"  Total Expected: {stats['total_expected']:.2f}\n")

            # Convergence indicator
            convergence_pct = abs(stats['deviation'] / stats['mean_expected'] * 100) if stats['mean_expected'] > 0 else 0
            text.insert(tk.END, f"\n  Convergence: ")
            if convergence_pct < 5:
                text.insert(tk.END, f"Excellent ({convergence_pct:.1f}% off) \u2713\n")
            elif convergence_pct < 10:
                text.insert(tk.END, f"Good ({convergence_pct:.1f}% off)\n")
            elif convergence_pct < 20:
                text.insert(tk.END, f"Moderate ({convergence_pct:.1f}% off)\n")
            else:
                text.insert(tk.END, f"Need more data ({convergence_pct:.1f}% off)\n")

            # Visual trend indicator
            text.insert(tk.END, f"\n  Luck Trend: ")
            if stats['deviation'] > stats['std_actual']:
                text.insert(tk.END, "Running hot! \U0001F525\n")
            elif stats['deviation'] < -stats['std_actual']:
                text.insert(tk.END, "Running cold \u2744\ufe0f\n")
            else:
                text.insert(tk.END, "Balanced \u2696\ufe0f\n")

    def clear_combat_results(self):
        """Clear combat results display"""
        self.combat_results_text.delete(1.0, tk.END)
        self.combat_results_text.insert(tk.END, "Combat results cleared.\n")

    def clear_combat_statistics(self):
        """Clear combat history and statistics"""
        self.combat_resolver.clear_history()
        self.combat_results_text.delete(1.0, tk.END)
        self.combat_results_text.insert(tk.END, "Combat statistics cleared.\nAll historical data has been reset.\n")

    def _increase_font_size(self):
        """Increase font size"""
        current = self.font_size.get()
        if current < 20:  # Max font size
            self.font_size.set(current + 1)
            self._update_font()

    def _decrease_font_size(self):
        """Decrease font size"""
        current = self.font_size.get()
        if current > 6:  # Min font size
            self.font_size.set(current - 1)
            self._update_font()

    def _reset_font_size(self):
        """Reset font size to default"""
        self.font_size.set(9)
        self._update_font()

    def _update_font(self):
        """Update font for all text widgets"""
        new_font = (self.font_family, self.font_size.get())
        self.results_text.configure(font=new_font)
        self.combat_results_text.configure(font=new_font)


def main():
    root = tk.Tk()
    app = DiceRollerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
