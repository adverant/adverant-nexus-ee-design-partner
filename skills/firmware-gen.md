---
name: firmware-gen
displayName: "Firmware Generator"
description: "Generate firmware scaffolding, HAL layers, and drivers for embedded MCUs"
version: 1.0.0
status: published
visibility: organization

allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Task

triggers:
  - /firmware-gen
  - /firmware
  - /mcu

capabilities:
  - name: generate
    description: Generate complete firmware project scaffolding
    parameters:
      - name: mcu_family
        type: string
        required: true
        description: "MCU family: stm32, esp32, ti_tms320, aurix, nrf, rpi_pico, imxrt"
      - name: mcu_part
        type: string
        required: true
        description: "Specific MCU part number (e.g., STM32H755ZI)"
      - name: rtos
        type: string
        required: false
        description: "RTOS: freertos, zephyr, tirtos, autosar, none"
      - name: features
        type: array
        required: false
        description: "Features to include: foc, uart, spi, i2c, can, usb, ethernet"
      - name: schematic_id
        type: string
        required: false
        description: "Schematic to extract pin mappings from"

  - name: hal
    description: Generate HAL (Hardware Abstraction Layer) code
    parameters:
      - name: peripherals
        type: array
        required: true
        description: List of peripherals to generate HAL for

  - name: driver
    description: Generate component driver from datasheet
    parameters:
      - name: component
        type: string
        required: true
        description: Component name or part number
      - name: datasheet_url
        type: string
        required: false
        description: URL to component datasheet
      - name: interface
        type: string
        required: true
        description: "Interface type: gpio, uart, spi, i2c, can"
---

# Firmware Generator

## Overview

Generate production-quality firmware scaffolding for embedded MCUs, including HAL layers, peripheral drivers, and RTOS configuration.

## Supported MCU Families

| Family | Parts | HAL Support | RTOS |
|--------|-------|-------------|------|
| **STM32** | F0, F1, F2, F3, F4, F7, H7, L0, L1, L4, G0, G4, U5, WB, WL | Full HAL + LL | FreeRTOS, Zephyr |
| **ESP32** | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C6 | ESP-IDF | FreeRTOS |
| **TI TMS320** | F28335, F28379D, F28069M | DriverLib | TI-RTOS |
| **Infineon AURIX** | TC2xx, TC3xx, TC4xx | iLLD | AUTOSAR |
| **Nordic nRF** | nRF52, nRF53, nRF54, nRF91 | nrfx | Zephyr |
| **Raspberry Pi** | RP2040, RP2350 | Pico SDK | FreeRTOS |
| **NXP i.MX RT** | RT1010, RT1050, RT1060, RT1170 | MCUXpresso | FreeRTOS, Zephyr |

## Usage

### Generate Complete Project

```bash
/firmware-gen --mcu=stm32h755 --rtos=freertos --features=foc,can,uart
```

### Generate from Schematic

```bash
/firmware-gen --mcu=stm32h755 --schematic=foc-esc.kicad_sch
```

This extracts pin mappings from the schematic and generates matching peripheral configurations.

### Generate HAL Only

```bash
/hal-gen --mcu=stm32h755 --peripherals=uart1,spi2,tim1,adc1
```

### Generate Driver

```bash
/driver-gen --component=DRV8323RS --interface=spi --datasheet=https://ti.com/...
```

## Generated Structure

```
firmware/
├── src/
│   ├── main.c                    # Application entry point
│   ├── app/                      # Application logic
│   │   ├── app.c
│   │   ├── app.h
│   │   └── tasks/                # RTOS tasks
│   │       ├── motor_task.c
│   │       ├── comm_task.c
│   │       └── safety_task.c
│   ├── hal/                      # Hardware abstraction
│   │   ├── hal_gpio.c
│   │   ├── hal_uart.c
│   │   ├── hal_spi.c
│   │   ├── hal_adc.c
│   │   ├── hal_pwm.c
│   │   └── hal_timer.c
│   ├── drivers/                  # Component drivers
│   │   ├── drv_mosfet_driver.c
│   │   ├── drv_current_sensor.c
│   │   └── drv_encoder.c
│   ├── foc/                      # Field-oriented control
│   │   ├── foc.c
│   │   ├── clarke_park.c
│   │   ├── svpwm.c
│   │   └── observer.c
│   └── bsp/                      # Board support package
│       ├── bsp.c
│       ├── bsp.h
│       └── pin_config.h          # Pin mappings from schematic
├── include/
│   └── config.h                  # Project configuration
├── lib/                          # External libraries
│   ├── cmsis/
│   └── hal/
├── startup/
│   └── startup_stm32h755.s
├── linker/
│   └── STM32H755ZI_FLASH.ld
├── CMakeLists.txt
├── Makefile
└── .clang-format
```

## FOC Motor Control Generation

For motor controller projects:

```bash
/firmware-gen --mcu=stm32h755 --features=foc,triple-redundant
```

Generates:
- **FOC Algorithm**: Clarke/Park transforms, SVPWM, current PI controllers
- **Observer**: Sensorless rotor position estimation
- **Safety**: Over-current, over-temp, phase loss detection
- **Redundancy**: Triple-MCU voting logic (for AURIX/TMS320/STM32H7)

## Triple-Redundant Architecture

For safety-critical applications:

```bash
/firmware-gen --mcu=aurix --features=foc,triple-redundant,lockstep
```

Generates:
- Primary MCU firmware (full control)
- Voter MCU firmware (comparison logic)
- Safety MCU firmware (watchdog)
- Inter-MCU communication protocol
- Byzantine fault tolerance voting

## HAL Layer Details

Each HAL module provides:

```c
// Example: hal_pwm.h
typedef struct {
    uint32_t frequency;
    uint16_t duty_cycle;
    uint8_t channel;
    bool center_aligned;
    bool complementary;
    uint16_t deadtime_ns;
} HAL_PWM_Config;

HAL_Status HAL_PWM_Init(HAL_PWM_Config* config);
HAL_Status HAL_PWM_SetDuty(uint8_t channel, uint16_t duty);
HAL_Status HAL_PWM_SetDuty3Phase(uint16_t a, uint16_t b, uint16_t c);
HAL_Status HAL_PWM_EnableOutputs(void);
HAL_Status HAL_PWM_DisableOutputs(void);
```

## Driver Generation

From datasheet analysis:

```bash
/driver-gen --component=DRV8323RS --interface=spi
```

Generates:
- Register definitions
- SPI communication functions
- Initialization sequence
- Status reading
- Fault handling
- Example usage code

## RTOS Configuration

### FreeRTOS

```c
// Generated FreeRTOSConfig.h highlights
#define configUSE_PREEMPTION            1
#define configCPU_CLOCK_HZ              480000000
#define configTICK_RATE_HZ              10000
#define configMAX_PRIORITIES            32
#define configMINIMAL_STACK_SIZE        256
#define configTOTAL_HEAP_SIZE           (128 * 1024)
```

### Zephyr

```yaml
# Generated prj.conf
CONFIG_HEAP_MEM_POOL_SIZE=65536
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_PWM=y
CONFIG_ADC=y
CONFIG_SPI=y
CONFIG_CAN=y
```

## Build System

CMake-based with presets:

```bash
# Configure
cmake --preset=release

# Build
cmake --build --preset=release

# Flash
cmake --build --preset=release --target flash
```

## Examples

### STM32H7 FOC ESC

```bash
/firmware-gen \
  --mcu=stm32h755zi \
  --rtos=freertos \
  --features=foc,can,uart,spi \
  --schematic=foc-esc.kicad_sch
```

### ESP32-S3 IoT Sensor

```bash
/firmware-gen \
  --mcu=esp32s3 \
  --rtos=freertos \
  --features=wifi,ble,i2c,spi
```

### AURIX Triple-Core Safety MCU

```bash
/firmware-gen \
  --mcu=tc397 \
  --rtos=autosar \
  --features=foc,lockstep,safe_wdg
```
