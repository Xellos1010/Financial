@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: Attach inline policy to GlueServiceRole
aws iam put-role-policy --role-name GlueServiceRole --policy-name GlueServiceRolePolicy --policy-document file://glue-service-role-policy.json --profile %AWS_PROFILE%

endlocal
pause
