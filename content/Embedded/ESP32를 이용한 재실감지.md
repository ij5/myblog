---
title: ESP32를 이용한 재실감지 센서 만들기
date: 2024-04-29
tags:
  - BLE
  - iBeacon
  - Bluetooth
  - Beacon
  - ESP32
  - ESPresense
authors:
  - 이재희
slug: esphome-presence
---
# 개요
몇 달 전에 산 ESP32를 쓸 곳이 별로 없어서 계속 방치하던 중 재실 감지 센서 아이디어가 떠올랐다.
원래 esp32를 재실감지용으로 사용하는 사람이 많아 [ESPresense](https://espresense.com)라는 굉장히 편리한 프로젝트가 있다. 이거 말고도 [ESPHome](https://esphome.io)이라는 프로젝트가 있는데, ESPHome은 좀 무거운 듯하고 와이파이 연결 에러가 자꾸 발생해서 그냥 ESPresense를 사용하기로 했다.

# ESPresense 설치
설치 방법은 굉장히 간단하다. 그냥 ESPresense 공식 사이트에 접속하면 웹페이지 내에서 크롬 시리얼 포트를 통해 esp32에 프로그램을 설치할 수 있다. 설치 후 와이파이 세팅까지 끝내면 내부망으로 esp32 웹서버에 접속할 수 있다.

# 문제 발생
생각해보니 esp32는 ble를 사용하여 스마트폰을 식별한다. 그런데 안드로이드는 아이폰과 다르게 ble로 식별할 수 없다. 따라서 별도의 앱을 설치해야 하는데, 이것때문에 빡쳐서 때려침
