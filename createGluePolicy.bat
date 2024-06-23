@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: Create the policy
aws iam create-policy --policy-name GlueFullAccessPolicy --policy-document file://GlueFullAccessPolicy.json --profile %AWS_PROFILE%

endlocal
pause