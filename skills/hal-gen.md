---
name: hal-gen
displayName: "HAL Generator"
description: "Generate Hardware Abstraction Layer code for embedded systems"
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
  - /hal-gen
  - /hal
  - /hardware-abstraction

capabilities:
  - name: generate
    description: Generate HAL layer for target MCU
    parameters:
      - name: mcu
        type: string
        required: true
        description: Target MCU part number
      - name: peripherals
        type: array
        required: true
        description: Peripherals to initialize
      - name: schematic_id
        type: string
        required: false
        description: Schematic for pin mapping

  - name: configure_peripheral
    description: Configure specific peripheral
    parameters:
      - name: peripheral
        type: string
        required: true
        description: "Peripheral: gpio, uart, spi, i2c, adc, timer"
      - name: instance
        type: string
        required: true
        description: Instance number
      - name: config
        type: object
        required: true
        description: Configuration parameters

  - name: generate_pinmux
    description: Generate pin multiplexer configuration
    parameters:
      - name: mcu
        type: string
        required: true
        description: Target MCU
      - name: assignments
        type: array
        required: true
        description: Pin-to-function assignments
---

# HAL Generator

## Overview

Generate production-ready Hardware Abstraction Layer code from schematic pin assignments. Supports all major MCU families with vendor-specific HAL integration.

## Usage

### Generate Complete HAL

```bash
# Generate from schematic
/hal-gen --mcu=STM32H755ZIT6 --schematic=sch-001

# Specify peripherals
/hal-gen --mcu=ESP32-WROOM-32E --peripherals=wifi,uart,gpio,adc

# With RTOS support
/hal-gen --mcu=STM32G474RET6 --rtos=freertos --peripherals=all
```

### Configure Peripherals

```bash
# UART configuration
/hal-gen uart --instance=1 --baud=115200 --parity=none --flow=none

# SPI configuration
/hal-gen spi --instance=1 --mode=master --speed=10MHz --bits=8 --cpol=0 --cpha=0

# I2C configuration
/hal-gen i2c --instance=1 --speed=400kHz --addressing=7bit

# ADC configuration
/hal-gen adc --instance=1 --resolution=12bit --channels=4 --dma=yes

# Timer/PWM configuration
/hal-gen timer --instance=2 --mode=pwm --freq=20kHz --channels=3
```

### Pin Mapping

```bash
# Generate from schematic
/hal-gen pinmux --schematic=sch-001

# Manual assignment
/hal-gen pinmux --mcu=STM32G4 --assign="PA0:UART1_TX,PA1:UART1_RX,PB0:ADC1_IN8"
```

## Supported MCU Families

| Family | HAL Type | Version |
|--------|----------|---------|
| STM32 | STM32 HAL/LL | 1.11.0 |
| ESP32 | ESP-IDF | 5.0 |
| TI C2000 | DriverLib | 2.0 |
| AURIX | iLLD | 1.0 |
| Nordic nRF | nrfx | 3.0 |
| RP2040 | Pico SDK | 1.5 |
| NXP i.MX RT | MCUXpresso | 2.0 |

## Generated Code Structure

```
generated/
├── hal/
│   ├── hal_gpio.c
│   ├── hal_gpio.h
│   ├── hal_uart.c
│   ├── hal_uart.h
│   ├── hal_spi.c
│   ├── hal_spi.h
│   ├── hal_i2c.c
│   ├── hal_i2c.h
│   ├── hal_adc.c
│   ├── hal_adc.h
│   ├── hal_timer.c
│   └── hal_timer.h
├── config/
│   ├── pin_config.h
│   ├── clock_config.h
│   └── peripheral_config.h
└── drivers/
    └── (component drivers)
```

## Code Example

### Generated UART HAL (STM32)

```c
// hal_uart.h
#ifndef HAL_UART_H
#define HAL_UART_H

#include "stm32h7xx_hal.h"

typedef enum {
    HAL_UART_1,
    HAL_UART_2,
    HAL_UART_3,
    HAL_UART_COUNT
} hal_uart_instance_t;

int hal_uart_init(hal_uart_instance_t instance);
int hal_uart_deinit(hal_uart_instance_t instance);
int hal_uart_transmit(hal_uart_instance_t instance, const uint8_t* data, size_t len, uint32_t timeout);
int hal_uart_receive(hal_uart_instance_t instance, uint8_t* data, size_t len, uint32_t timeout);
int hal_uart_transmit_dma(hal_uart_instance_t instance, const uint8_t* data, size_t len);
int hal_uart_receive_dma(hal_uart_instance_t instance, uint8_t* data, size_t len);

#endif
```

### Portable API

All HAL functions follow consistent patterns:
- Return codes (0 = success, negative = error)
- Timeout parameters where applicable
- DMA variants for bulk transfers
- Interrupt callbacks

## API Endpoint

```
POST /ee-design/api/v1/firmware/hal
```

Request body:
```json
{
  "mcu": "STM32H755ZIT6",
  "schematicId": "sch-uuid",
  "peripherals": [
    {"type": "uart", "instance": "1", "config": {"baudRate": 115200}},
    {"type": "spi", "instance": "1", "config": {"mode": "master", "speed": 10000000}},
    {"type": "gpio", "pins": ["PA0", "PA1", "PB5"]}
  ],
  "options": {
    "generateTests": true,
    "dmaSupport": true,
    "interruptCallbacks": true
  }
}
```

## Integration

Part of EE Design Partner Phase 7 (Firmware). Generates from:
- Schematic pin assignments
- Architecture specifications
- Peripheral requirements

Output feeds into:
- Driver generation
- Application scaffolding
- Test framework
