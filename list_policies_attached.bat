@echo off
setlocal

set AWS_PROFILE=SystemDeveloper-Xellos

:: List policies attached to GlueServiceRole
aws iam list-attached-role-policies --role-name GlueServiceRole --profile %AWS_PROFILE%
aws iam list-role-policies --role-name GlueServiceRole --profile %AWS_PROFILE%

:: List policies attached to SystemDeveloper-Xellos user
aws iam list-attached-user-policies --user-name SystemDeveloper-Xellos --profile %AWS_PROFILE%
aws iam list-user-policies --user-name SystemDeveloper-Xellos --profile %AWS_PROFILE%

endlocal
pause
