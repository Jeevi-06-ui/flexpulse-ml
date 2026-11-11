# Phase 2 Documentation: Real Data Acquisition, Serial Telemetry & Canonical Storage

This document provides the scientific and electrical engineering documentation for **Phase 2 of FlexPulse-ML**.

---

## 1. How the MacBook Communicates with the Controller

```
+--------------------------+                         +-------------------------------+
|  MacBook (Apple Silicon) |                         | ESP32 + INA226 Test Fixture   |
|                          |    USB-UART (CDC-ACM)   |                               |
|  [flexpulse CLI / HAL]   |<=======================>|  [ESP32 Microcontroller]      |
|    - PySerial Adapter    |   115200 Baud, 8-N-1    |    - I2C Driver               |
|    - NDJSON Parser       |                         |    - INA226 Bi-Directional    |
|    - Scientific Validator|                         |      Current/Power Monitor    |
+--------------------------+                         +-------------------------------+
            |                                                        |
            v                                                        v
   [data/raw/real/ CSV]                                     [USB-C Cable Under Test]
```

### Communication Pipeline:
1. **Physical Layer:** The ESP32 is connected to the MacBook via a USB-C/USB-A data cable. macOS exposes this device as a character device node (typically `/dev/tty.usbserial-*` or `/dev/cu.wchusbserial*`).
2. **Transport Layer:** PySerial opens this port at 115,200 baud with 8 data bits, no parity, and 1 stop bit (8-N-1). Flow control (`rtscts` and `dsrdtr`) is disabled to prevent accidental resets of the ESP32.
3. **Application Protocol:** The controller transmits **newline-delimited JSON (NDJSON)** packets. Each packet is a self-contained JSON string terminated by `\n`.
4. **Safety Enforcement:** The driver is **read-only**. It never issues commands to alter programmable power supply rails, activate switching relays, or toggle load setpoints. Safety limits are independently governed by the hardware test fixture.

---

## 2. Documented Example ESP32 Packet Format

The ESP32 transmits telemetry in the following JSON structure:

```json
{
  "packet_type": "TELEMETRY",
  "sample_idx": 142,
  "timestamp_ms": 1420.5,
  "vin_mv": 5012,
  "vout_mv": 4892,
  "current_ma": 1502.4,
  "temp_c": 26.4,
  "settled": true,
  "quality": "OK",
  "crc16": "A1F4"
}
```

### Telemetry Packet Fields:
* `packet_type`: Identifies packet class (`"TELEMETRY"`, `"HEARTBEAT"`, or `"CALIBRATION"`).
* `sample_idx`: 32-bit unsigned incremental integer. Used by the MacBook to detect dropped or out-of-order packets.
* `timestamp_ms`: Milliseconds elapsed on the ESP32 hardware timer since trial start.
* `vin_mv`: Upstream source voltage measured before the cable, in millivolts ($5012\text{ mV} = 5.012\text{ V}$).
* `vout_mv`: Downstream sink voltage measured at the load, in millivolts ($4892\text{ mV} = 4.892\text{ V}$).
* `current_ma`: Bus current passing through the shunt resistor, in milliamperes ($1502.4\text{ mA} = 1.5024\text{ A}$).
* `temp_c`: Ambient/connector temperature measured via thermistor or onboard sensor in Celsius.
* `settled`: Boolean flag (`true`/`false`). True when the INA226 ADC conversion time has elapsed and the power path is in steady state.
* `quality`: Diagnostic flag (`"OK"`, `"CLIPPED"`, `"JITTER_HIGH"`, `"DISCONNECTED"`).
* `crc16`: Optional cyclic redundancy check for noisy laboratory bench environments.

---

## 3. How the Incoming Measurement Becomes a Canonical CSV Row

When raw bytes arrive at the MacBook:
1. **Line Buffering & UTF-8 Decoding:** PySerial reads until `\n` and decodes using UTF-8 (handling corrupted replacement characters).
2. **Syntax & Key Verification:** The string is parsed via Python's `json.loads()`. If syntax is corrupt, the line is quarantined and `stats.corrupt_json_lines` is incremented.
3. **Physical Bound Verification:**
   * Voltages must be in $[0.0, 28.0]\text{ V}$.
   * Currents must be in $[0.0, 6.0]\text{ A}$.
   * Passive power rule: $V_{\text{out}} \le V_{\text{in}} + 0.15\text{ V}$ (allowing $150\text{ mV}$ for measurement noise).
4. **Metadata Merging:** The observation is merged with the active trial's registered metadata (`cable_id`, `session_id`, `trial_id`, `ground_truth_label`, `fixture_id`, etc.).
5. **Canonical CSV Serialization:** The record is written as a standardized 22-column row preserving raw numeric values without lossy floating-point rounding.

---

## 4. The 22-Column Canonical Schema Reference

| Column Index | Field Name | Data Type | Units / Valid Values | Description |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `timestamp_utc` | String | ISO 8601 UTC | Exact host timestamp when record was logged (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`). |
| 2 | `cable_id` | String | e.g. `CABLE-ANK-001` | Unique physical identifier of the cable under test. Critical for Leave-One-Cable-Out CV. |
| 3 | `session_id` | String | e.g. `SESS-20261111-01` | Experimental session grouping. |
| 4 | `trial_id` | String | e.g. `TR-001` | Specific experimental trial pass. |
| 5 | `sample_index` | Integer | $\ge 0$ | Monotonic sample index within the trial. |
| 6 | `vin_v` | Float | Volts ($[0.0, 28.0]$) | Source voltage upstream of the cable under test. |
| 7 | `vout_v` | Float | Volts ($[0.0, 28.0]$) | Sink voltage downstream of the cable at the electronic load. |
| 8 | `current_a` | Float | Amperes ($[0.0, 6.0]$) | Measured DC current passing through the cable conductor. |
| 9 | `load_setpoint_a` | Float | Amperes | Target current requested by the electronic load controller. |
| 10 | `temperature_c` | Float | ${}^\circ\text{C}$ | Ambient or connector temperature. |
| 11 | `source_type` | String | `REAL_MEASUREMENT` or `SYNTHETIC_DEMO` | Scientific authenticity tag. Strictly isolated. |
| 12 | `ground_truth_label` | String | `INTACT`, `RESISTIVE_BEND`, etc. | Physical condition of the cable. |
| 13 | `ground_truth_source` | String | `HUMAN_VERIFIED`, etc. | Provenance of the label assignment. |
| 14 | `fault_origin` | String | `NATURAL_WEAR`, `FIXTURE_INJECTED`, `GOLDEN_REFERENCE` | How the fault condition was produced. |
| 15 | `fault_condition` | String | e.g. `SERIES_100M_RESISTOR`, `NONE` | Detailed engineering description of the fault injection. |
| 16 | `fixture_id` | String | e.g. `FIXTURE_ESP32_INA226_01` | Identifier of the test fixture hardware. |
| 17 | `instrument_id` | String | e.g. `INA226_I2C_ADDR_0x40` | Specific ADC/sensor channel. |
| 18 | `calibration_id` | String | e.g. `CAL_20261111_01` | Fixture tare resistance calibration profile. |
| 19 | `action_id` | String | e.g. `STATIC_HOLD`, `FLEX_SWEEP_0_90` | Mechanical or electrical test action being executed. |
| 20 | `motion_region` | String | `STATIC`, `PLUG_A_STRAIN_RELIEF`, `CABLE_MIDSPAN` | Physical flex point on the cable. |
| 21 | `is_settled` | Boolean | `True` / `False` | True if ADC settling period elapsed; False during transient load switching. |
| 22 | `measurement_quality` | String | `OK`, `CLIPPED`, `JITTER_HIGH`, `SUSPECT` | Sensor health and acquisition integrity indicator. |

---

## 5. How Real Measurements Differ from Simulated Examples

| Dimension | Real Measurements (`REAL_MEASUREMENT`) | Simulated Examples (`SYNTHETIC_DEMO`) |
| :--- | :--- | :--- |
| **Origin** | Physical copper conductors, USB-C connectors, INA226 ADC conversion. | Mathematical model in [`synthetic_generator.py`](file:///Users/jeevithp/Documents/ML%20Research%20paper/flexpulse_ml/acquisition/synthetic_generator.py). |
| **Storage Location** | `data/raw/real/` | `data/raw/synthetic_demo/` or `data/demo/` |
| **Source Type Tag** | `source_type = "REAL_MEASUREMENT"` | `source_type = "SYNTHETIC_DEMO"` |
| **Noise Profile** | True physical noise: 1/f flicker noise, contact micro-arcing, Johnson-Nyquist thermal noise, and ADC quantisation. | Ideal Gaussian noise ($\mathcal{N}(0, \sigma^2)$) superimposed on deterministic curve. |
| **Use in Research Paper** | Mandatory for final model training, LOCO benchmarks, and reported F1-scores. | **Strictly prohibited** from research benchmarks. Used only for software unit tests and pipeline validation. |
| **Integrity Enforcement** | Disallowing synthetic data immediately quarantines any synthetic row. | Tagged automatically; rejected if mixed into real benchmarks. |

---

## 6. Guide: Connecting the Physical ESP32 Later

When your Presidency University lab hardware is assembled:
1. **Connect Hardware:** Plug the ESP32 USB interface into your MacBook USB port.
2. **Find Serial Node:**
   ```bash
   ls /dev/tty.usb* /dev/cu.usb*
   ```
   Example output: `/dev/tty.usbserial-0001` or `/dev/cu.wchusbserial110`.
3. **Configure Environment:**
   Edit `.env` (or set the environment variable):
   ```bash
   FLEXPULSE_SERIAL_PORT=/dev/tty.usbserial-0001
   FLEXPULSE_SERIAL_BAUD=115200
   ```
4. **Register Your Physical Cable:**
   ```bash
   flexpulse register-cable --id CABLE-TEST-01 --manufacturer Anker --type USB_C_TO_C --current 3.0 --length 1.0
   ```
5. **Verify Hardware Connection:**
   Run the CLI status command to confirm the hardware handshake:
   ```bash
   flexpulse status
   ```
   The HAL will automatically establish the serial stream, validate incoming packets, and begin recording to `data/raw/real/`.
