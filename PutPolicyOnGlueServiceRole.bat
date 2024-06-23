@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: Update the GlueServiceRole with the necessary permissions
aws iam put-role-policy --role-name GlueServiceRole --policy-name GluePolicy --policy-document file://glue-policy.json --profile %AWS_PROFILE%

endlocal
pause