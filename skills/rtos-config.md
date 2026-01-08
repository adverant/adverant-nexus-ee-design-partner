---
name: rtos-config
displayName: "RTOS Configurator"
description: "Configure and generate RTOS project scaffolding"
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
  - /rtos-config
  - /rtos
  - /configure-rtos

capabilities:
  - name: configure
    description: Configure RTOS settings
    parameters:
      - name: rtos
        type: string
        required: true
        description: "RTOS: freertos, zephyr, tirtos, autosar"
      - name: mcu
        type: string
        required: true
        description: Target MCU

  - name: add_task
    description: Add task to RTOS project
    parameters:
      - name: name
        type: string
        required: true
        description: Task name
      - name: priority
        type: number
        required: true
        description: Task priority
      - name: stack_size
        type: number
        required: false
        description: Stack size in bytes

  - name: analyze
    description: Analyze RTOS configuration
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
---

# RTOS Configurator

## Overview

Configure and generate RTOS projects with proper task structure, memory allocation, and synchronization primitives.

## Usage

### Configure RTOS

```bash
# FreeRTOS setup
/rtos-config freertos --mcu=STM32H755 --heap=32KB --tick=1000Hz

# Zephyr setup
/rtos-config zephyr --mcu=nRF52840 --scheduler=preemptive

# TI-RTOS setup
/rtos-config tirtos --mcu=TMS320F28379D
```

### Add Tasks

```bash
# Add task
/rtos-config add-task MainTask --priority=2 --stack=2048 --period=100ms

# Add ISR-deferred task
/rtos-config add-task CanRxTask --priority=4 --stack=1024 --isr-deferred

# Add idle hook
/rtos-config add-hook idle --function=EnterLowPower
```

### Analyze Configuration

```bash
# Stack usage analysis
/rtos-config analyze --project=proj-001 --type=stack

# CPU utilization estimate
/rtos-config analyze --project=proj-001 --type=cpu

# Memory usage
/rtos-config analyze --project=proj-001 --type=memory
```

## Supported RTOS

| RTOS | MCU Support | Features |
|------|-------------|----------|
| FreeRTOS | All ARM | Tasks, queues, semaphores |
| Zephyr | nRF, STM32, ESP32 | Full POSIX, BLE stack |
| TI-RTOS | C2000, MSP432 | Real-time analysis |
| AUTOSAR | AURIX, MPC | Automotive certified |

## Configuration Example

```yaml
# FreeRTOS Configuration
rtos:
  type: freertos
  version: 10.4.6
  config:
    tick_rate_hz: 1000
    max_priorities: 7
    heap_size: 32768
    minimal_stack: 128
    use_preemption: true
    use_mutexes: true
    use_counting_semaphores: true
    use_timers: true

tasks:
  - name: MainTask
    priority: 2
    stack_size: 2048
    period_ms: 100

  - name: CommTask
    priority: 3
    stack_size: 1024
    period_ms: 10

  - name: SensorTask
    priority: 2
    stack_size: 512
    period_ms: 50
```

## API Endpoint

```
POST /ee-design/api/v1/firmware/rtos
```

## Integration

Part of EE Design Partner Phase 7 (Firmware). Works with:
- HAL generation
- Task code templates
- Memory analysis
