# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_lattice_rnn.py

import numpy as np
from datetime import datetime, timedelta
import math
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd # For potential data handling



# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/quantum_lattice_rnn.py
# quantum_lattice_rnn.py

import numpy as np
from datetime import datetime, timedelta
import math
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 1. Utility Functions
# ----------------------------

def get_lunar_phase(date):
    epoch = datetime(2000, 1, 6)
    delta = date - epoch
    lunar_cycle = 29.530588
    return (delta.days % lunar_cycle) / lunar_cycle

def get_solar_cycle_influence(date):
    base_year = 2000
    years_since = date.year - base_year
    return 1 + 0.05 * math.sin(2 * math.pi * years_since / 11)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def text_to_seed_vector(text):
    return np.array([sum(ord(c) for c in text) % 1000, len(text) % 100]) if text else np.zeros(2)

def list_to_fourier_vector(seq):
    return np.array([1.0 + np.var(seq) / 100.0]) if len(seq) >= 2 else np.array([1.0])

# ----------------------------
# 2. Collapse History Manager
# ----------------------------

class CollapseHistoryManager:
    def __init__(self, history):
        self.history = sorted(history, key=lambda x: x['date'])

    def get_last_n_collapses(self, n):
        dates = sorted(set(r['date'] for r in self.history), reverse=True)[:n]
        return [r for r in self.history if r['date'] in dates]

    def get_frequency_score(self, num, days=1825):
        now = datetime.now()
        filtered = [r['number'] for r in self.history if (now - r['date']).days <= days]
        counts = Counter(filtered)
        return counts.get(num, 0) / max(1, sum(counts.values()))

    def get_days_since_last_seen(self, num):
        for r in reversed(self.history):
            if r['number'] == num:
                return (datetime.now() - r['date']).days
        return 9999

    def get_match_score(self, num, world, match_type):
        score = 0.0
        for r in world:
            if match_type == 'last2' and num[-2:] == r['number'][-2:]:
                score += 1.0
            elif match_type == 'front3' and num[:3] == r['number'][:3]:
                score += 1.0
        return score

    def get_twin_digit_match(self, num):
        counts = Counter(num)
        if any(c >= 3 for c in counts.values()): return 1.0
        if any(c == 2 for c in counts.values()): return 0.5
        return 0.0

# ----------------------------
# 3. Quantum Lattice RNN
# ----------------------------

class QuantumLatticeRNN:
    def __init__(self, manager, time_steps=5, feature_dim=8):
        self.hm = manager
        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.model = self._build_model()
        self.scaler = MinMaxScaler()

    def _build_model(self):
        model = Sequential([
            LSTM(128, input_shape=(self.time_steps, self.feature_dim)),
            Dense(64, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _prepare_data(self):
        sequences, targets = [], []
        unique_dates = sorted(set(r['date'] for r in self.hm.history))
        if len(unique_dates) <= self.time_steps: return np.array([]), np.array([])

        for i in range(len(unique_dates) - self.time_steps):
            seq, valid = [], True
            for d in unique_dates[i:i + self.time_steps]:
                records = [r for r in self.hm.history if r['date'] == d]
                if not records:
                    valid = False
                    break
                nums = [int(r['number']) for r in records]
                avg = np.mean(nums)
                freq = len(nums)
                features = [
                    avg / 1000.0, freq / 10.0,
                    get_lunar_phase(d), get_solar_cycle_influence(d),
                    self.hm.get_twin_digit_match(records[0]['number']),
                    1 if any(self.hm.get_twin_digit_match(r['number']) > 0 for r in records) else 0,
                    self.hm.get_match_score('000', records, 'last2'),
                    self.hm.get_match_score('000', records, 'front3')
                ]
                seq.append(features)
            if valid:
                next_day = unique_dates[i + self.time_steps]
                next_nums = [int(r['number']) for r in self.hm.history if r['date'] == next_day]
                if next_nums:
                    sequences.append(seq)
                    targets.append(np.mean(next_nums) / 1000.0)
        if not sequences: return np.array([]), np.array([])
        X = np.array(sequences)
        y = np.array(targets)
        orig_shape = X.shape
        X_reshaped = X.reshape(-1, self.feature_dim)
        X_scaled = self.scaler.fit_transform(X_reshaped).reshape(orig_shape)
        return X_scaled, y

    def train_rnn(self, epochs=50, batch=32):
        X, y = self._prepare_data()
        if X.size > 0:
            self.model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0)

    def predict_next_field_potential(self, recent_worlds):
        if len(recent_worlds) < self.time_steps:
            return 0.5
        seq = []
        for r in recent_worlds[-self.time_steps:]:
            nums = [rec['number'] for rec in recent_worlds if rec['date'] == r['date']]
            avg = np.mean([int(n) for n in nums])
            features = [
                avg / 1000.0, len(nums) / 10.0,
                get_lunar_phase(r['date']), get_solar_cycle_influence(r['date']),
                self.hm.get_twin_digit_match(nums[0] if nums else '000'),
                1 if any(self.hm.get_twin_digit_match(n) > 0 for n in nums) else 0,
                self.hm.get_match_score('000', recent_worlds, 'last2'),
                self.hm.get_match_score('000', recent_worlds, 'front3')
            ]
            seq.append(features)
        X = np.array([seq])
        reshaped = X.reshape(-1, self.feature_dim)
        scaled = self.scaler.transform(reshaped).reshape(X.shape)
        return sigmoid(self.model.predict(scaled)[0][0] * 5)



class QuantumLatticeField:
    def __init__(self, history_manager, user_seed=None, rnn_predictor=None):
        self.history_manager = history_manager
        self.user_seed = user_seed
        self.rnn_predictor = rnn_predictor

    def generate_field(self):
        # สร้าง field ข้อมูลจำลอง 1 ล้านหมายเลข x 12 มิติ
        return np.random.rand(1000000, 12)

    def collapse_field(self, field, top_n=5):
        # ใช้คอลัมน์แรกเป็น score จำลองในการจัดอันดับ
        scores = field[:, 0]
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [(str(i).zfill(6), scores[i]) for i in top_indices]



# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    print("🚀 Quantum Lattice RNN Ready for Prediction")
