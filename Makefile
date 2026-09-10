.PHONY: check artifacts extension-test report

check:
	python3 scripts/check_repository.py

artifacts:
	python3 scripts/verify_artifacts.py

extension-test:
	cd extension && npm ci && npm test

report:
	$(MAKE) -C report
