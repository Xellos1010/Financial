@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos
set ROLE_NAME=GlueServiceRole

:: Update the trust relationship for the role
aws iam update-assume-role-policy --role-name %ROLE_NAME% --policy-document file://trust-policy.json --profile %AWS_PROFILE%

endlocal
pause
