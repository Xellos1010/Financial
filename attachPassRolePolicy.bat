@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: Attach PassRole policy to the user
aws iam put-user-policy --user-name SystemDeveloper-Xellos --policy-name PassRolePolicy --policy-document file://iam-pass-role-policy.json --profile %AWS_PROFILE%

endlocal
pause
