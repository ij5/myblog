---
title: React Native BLE 라이브러리 정리
date: 2024-04-28
tags:
  - ReactNative
  - Beacon
  - iBeacon
  - Bluetooth
  - BLE
  - React
authors:
  - 이재희
slug: ble
---
리액트 네이티브 BLE(Bluetooth Low Energy) 라이브러리가 여러 개가 있는데, 그 중 사용해본 라이브러리를 설명하겠다. 아래는 깃허브 스타 수를 기준으로 정렬하였다.
# react-native-ble-plx

![](https://i.imgur.com/KB69qeh.png)

이 라이브러리는 크로스 플랫폼(Android, iOS)을 지원하여 개발자가 블루투스 서비스를 사용하기 쉽게 만들어 놓은 라이브러리다. 현재 깃허브 스타 수가 제일 많고, 기여가 활발한 편이다.
expo를 지원해서, `expo install` 커맨드로 설치하면 `AndroidManifest.xml`이나 `build.gradle` 등의 네이티브 파일을 수정하지 않아도 된다.

다만 사용 중 모든 BLE 디바이스가 검색이 안되는 문제가 발생하여 대략 3일동안 삽질했는데, 아래 이슈를 보고 해결하였다. 
[#1014](https://github.com/dotintent/react-native-ble-plx/issues/1014)
안드로이드 13 이상에서는 `neverForLocation`([링크](https://developer.android.com/develop/connectivity/bluetooth/bt-permissions?hl=ko#assert-never-for-location)) 권한 플래그가 새로 생겼는데, 이 권한 플래를 `android:usesPermissionFlags`에 포함하면 일부 BLE 비콘이 검색 결과에서 필터링된다. 나는 `android/app/src/AndroidManifest.xml` 파일에 직접 플래그를 포함하지 않았지만, 라이브러리 자체에서 플래그를 포함했기 때문에 비콘이 검색 결과에 뜨지 않았던 것이다. 따라서 `AndroidManifest.xml` 파일의 권한 부분에 다음과 같이  tools:remove 속성을 추가해야 한다.
```xml
<uses-permission android:name="android.permission.BLUETOOTH_SCAN" tools:remove="android:usesPermissionFlags" />
```

# react-native-ble-manager
`react-native-ble-plx`와 비슷하지만, BLE 스캔 방식이 더 불편하다고 느꼈다. 예를 들어 디바이스 스캔 후 결과값을 출력하려면 RN의 네이티브 모듈에 이벤트 리스너를 추가해야 한다.
# react-native-beacons-manager
기여가 거의 없는 수준이라 추천하지는 않는다.
