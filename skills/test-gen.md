---
name: test-gen
displayName: "Test Generator"
description: "Generate unit tests, integration tests, and test procedures for firmware"
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
  - /test-gen
  - /generate-tests
  - /test

capabilities:
  - name: unit_tests
    description: Generate unit tests for firmware modules
    parameters:
      - name: source_file
        type: string
        required: true
        description: Source file to test
      - name: framework
        type: string
        required: false
        description: "Framework: unity, cpputest, gtest"

  - name: integration_tests
    description: Generate integration test suite
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: modules
        type: array
        required: true
        description: Modules to test

  - name: hardware_tests
    description: Generate hardware test procedures
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: type
        type: string
        required: true
        description: "Test type: power_on, functional, environmental"

  - name: coverage
    description: Analyze test coverage
    parameters:
      - name: project_id
        type: string
        required: true
        description: Project identifier
      - name: report_format
        type: string
        required: false
        description: "Format: html, xml, lcov"
---

# Test Generator

## Overview

Generate comprehensive test suites for embedded firmware including unit tests, integration tests, and hardware test procedures. Supports multiple test frameworks and provides coverage analysis.

## Usage

### Unit Tests

```bash
# Generate tests for single file
/test-gen unit --file=src/drivers/motor_driver.c --framework=unity

# Generate tests for module
/test-gen unit --module=hal --framework=cpputest

# All project tests
/test-gen unit --project=proj-001 --all
```

### Integration Tests

```bash
# Module integration
/test-gen integration --modules=hal,drivers,app

# Communication tests
/test-gen integration --interfaces=uart,can,spi

# System tests
/test-gen integration --system --endpoints=all
```

### Hardware Tests

```bash
# Power-on self-test
/test-gen hardware post --project=proj-001

# Functional tests
/test-gen hardware functional --project=proj-001 --coverage=100%

# Environmental tests
/test-gen hardware environmental --temp="-40,125" --humidity="10,90"
```

### Coverage Analysis

```bash
# Run coverage
/test-gen coverage --project=proj-001 --threshold=80%

# Gap analysis
/test-gen coverage gaps --project=proj-001
```

## Supported Frameworks

| Framework | Language | Best For |
|-----------|----------|----------|
| Unity | C | Small, embedded |
| CppUTest | C/C++ | TDD, mocking |
| Google Test | C++ | Large projects |
| pytest | Python | HIL testing |

## Generated Test Structure

```
tests/
├── unit/
│   ├── test_hal_gpio.c
│   ├── test_hal_uart.c
│   ├── test_motor_driver.c
│   └── test_pid_controller.c
├── integration/
│   ├── test_communication.c
│   ├── test_power_sequence.c
│   └── test_safety_monitor.c
├── hardware/
│   ├── post_procedure.md
│   ├── functional_tests.md
│   └── environmental_tests.md
├── mocks/
│   ├── mock_hal.c
│   └── mock_peripherals.c
└── runners/
    ├── test_runner.c
    └── CMakeLists.txt
```

## Test Example

### Generated Unit Test (Unity)

```c
#include "unity.h"
#include "hal_uart.h"
#include "mock_stm32_hal.h"

void setUp(void) {
    mock_uart_reset();
}

void tearDown(void) {
    // Cleanup
}

void test_uart_init_should_configure_peripheral(void) {
    // Arrange
    mock_uart_expect_init(UART1, 115200, UART_WORDLENGTH_8B);

    // Act
    int result = hal_uart_init(HAL_UART_1);

    // Assert
    TEST_ASSERT_EQUAL(0, result);
    mock_uart_verify();
}

void test_uart_transmit_should_send_data(void) {
    // Arrange
    uint8_t data[] = {0x01, 0x02, 0x03};
    mock_uart_expect_transmit(UART1, data, 3);

    // Act
    int result = hal_uart_transmit(HAL_UART_1, data, 3, 100);

    // Assert
    TEST_ASSERT_EQUAL(0, result);
}

void test_uart_receive_timeout_should_return_error(void) {
    // Arrange
    uint8_t buffer[10];
    mock_uart_set_timeout();

    // Act
    int result = hal_uart_receive(HAL_UART_1, buffer, 10, 100);

    // Assert
    TEST_ASSERT_EQUAL(-1, result);
}
```

## Coverage Report

```
Code Coverage Report - proj-001
================================
Module          | Lines  | Branches | Functions
----------------|--------|----------|----------
hal_gpio        | 95%    | 88%      | 100%
hal_uart        | 92%    | 85%      | 100%
hal_spi         | 78%    | 72%      | 90%
motor_driver    | 88%    | 80%      | 95%
pid_controller  | 100%   | 95%      | 100%
----------------|--------|----------|----------
Total           | 91%    | 84%      | 97%
```

## Hardware Test Procedures

### POST (Power-On Self-Test)

```markdown
# Power-On Self-Test Procedure

## 1. Power Sequencing
- [ ] Apply 12V input
- [ ] Verify 3.3V rail (3.3V ±5%)
- [ ] Verify 5V rail (5.0V ±5%)
- [ ] Measure inrush current (<2A)

## 2. MCU Check
- [ ] LED1 blinks at 1Hz
- [ ] UART console responds
- [ ] CAN bus ACK

## 3. Peripheral Verification
- [ ] ADC self-test pass
- [ ] SPI loopback pass
- [ ] I2C scan finds devices
```

## API Endpoint

```
POST /ee-design/api/v1/testing/generate
```

Request body:
```json
{
  "projectId": "proj-uuid",
  "type": "unit",
  "framework": "unity",
  "options": {
    "generateMocks": true,
    "coverageThreshold": 80,
    "includeEdgeCases": true
  }
}
```

## Integration

Part of EE Design Partner Phase 8 (Testing). Provides:
- Automated test generation
- Coverage tracking
- Regression test suite
- Hardware test documentation
