# Consumer contract snapshots

이 디렉터리의 schema는 `customs-data-platform`이 소유한 canonical 계약의 pinned consumer
snapshot이다. AI Agent가 계약 원본을 소유한다는 의미가 아니며 임의 수정하지 않는다. 갱신 시 provider
release/revision과 SHA-256을 lock manifest에 기록하고 회귀 테스트를 통과시킨다.
