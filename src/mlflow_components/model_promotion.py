"""
MLflow model promotion workflow and stage management.
Handles automated model promotion between stages with validation.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from mlflow.entities.model_registry import ModelVersion
from .model_registry import get_model_registry

logger = logging.getLogger(__name__)


class ModelPromotionWorkflow:
    """
    Manages model promotion workflow between stages.
    Provides automated promotion with validation and rollback capabilities.
    """
    
    def __init__(self):
        """Initialize model promotion workflow."""
        self.registry = get_model_registry()
        self.validation_functions: Dict[str, List[Callable]] = {
            'staging': [],
            'production': [],
        }
    
    def add_validation_function(self, stage: str, validation_func: Callable) -> None:
        """
        Add validation function for a specific stage.
        
        Args:
            stage: Target stage ("staging", "production")
            validation_func: Function that takes ModelVersion and returns (bool, str)
        """
        if stage not in self.validation_functions:
            self.validation_functions[stage] = []
        
        self.validation_functions[stage].append(validation_func)
        logger.info(f"Added validation function for stage '{stage}'")
    
    def validate_model_for_stage(self, model_version: ModelVersion, stage: str) -> tuple[bool, List[str]]:
        """
        Validate model version for promotion to specific stage.
        
        Args:
            model_version: Model version to validate
            stage: Target stage
            
        Returns:
            Tuple of (is_valid, list_of_messages)
        """
        validation_results = []
        all_valid = True
        
        # Run all validation functions for the stage
        for validation_func in self.validation_functions.get(stage, []):
            try:
                is_valid, message = validation_func(model_version)
                validation_results.append(message)
                if not is_valid:
                    all_valid = False
            except Exception as e:
                error_msg = f"Validation function failed: {e}"
                validation_results.append(error_msg)
                all_valid = False
                logger.error(error_msg)
        
        return all_valid, validation_results
    
    def promote_model(self, name: str, version: str, target_stage: str,
                     archive_existing: bool = True,
                     skip_validation: bool = False) -> Dict[str, Any]:
        """
        Promote model version to target stage with validation.
        
        Args:
            name: Model name
            version: Model version
            target_stage: Target stage ("staging", "production")
            archive_existing: Whether to archive existing versions in target stage
            skip_validation: Whether to skip validation checks
            
        Returns:
            Promotion result dictionary
        """
        try:
            # Get model version
            model_version = self.registry.get_model_version(name, version)
            if not model_version:
                return {
                    'success': False,
                    'error': f'Model version {name} v{version} not found'
                }
            
            # Validate model for target stage
            if not skip_validation:
                is_valid, validation_messages = self.validate_model_for_stage(model_version, target_stage)
                if not is_valid:
                    return {
                        'success': False,
                        'error': 'Model validation failed',
                        'validation_messages': validation_messages
                    }
            
            # Get current stage for rollback info
            current_stage = model_version.current_stage
            
            # Perform promotion
            updated_version = self.registry.transition_model_version_stage(
                name=name,
                version=version,
                stage=target_stage,
                archive_existing_versions=archive_existing
            )
            
            # Add promotion metadata
            promotion_tags = {
                'promoted_at': datetime.utcnow().isoformat(),
                'promoted_from': current_stage,
                'promoted_to': target_stage,
                'promotion_type': 'automated' if not skip_validation else 'manual'
            }
            
            self.registry.update_model_version(
                name=name,
                version=version,
                tags=promotion_tags
            )
            
            logger.info(f"Successfully promoted model '{name}' v{version} from '{current_stage}' to '{target_stage}'")
            
            return {
                'success': True,
                'model_name': name,
                'version': version,
                'previous_stage': current_stage,
                'new_stage': target_stage,
                'promotion_timestamp': promotion_tags['promoted_at'],
                'validation_messages': validation_messages if not skip_validation else []
            }
            
        except Exception as e:
            error_msg = f"Failed to promote model '{name}' v{version} to '{target_stage}': {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def rollback_model(self, name: str, target_stage: str) -> Dict[str, Any]:
        """
        Rollback to previous model version in target stage.
        
        Args:
            name: Model name
            target_stage: Stage to rollback in
            
        Returns:
            Rollback result dictionary
        """
        try:
            # Get all versions in the target stage, sorted by creation time
            versions = self.registry.search_model_versions(
                filter_string=f"name='{name}' and current_stage='{target_stage}'"
            )
            
            if len(versions) < 2:
                return {
                    'success': False,
                    'error': f'Not enough versions in stage {target_stage} for rollback'
                }
            
            # Sort by creation timestamp (newest first)
            versions.sort(key=lambda v: v.creation_timestamp, reverse=True)
            
            current_version = versions[0]
            previous_version = versions[1]
            
            # Archive current version
            self.registry.transition_model_version_stage(
                name=name,
                version=current_version.version,
                stage='archived'
            )
            
            # Promote previous version back to target stage
            self.registry.transition_model_version_stage(
                name=name,
                version=previous_version.version,
                stage=target_stage
            )
            
            # Add rollback metadata
            rollback_tags = {
                'rolled_back_at': datetime.utcnow().isoformat(),
                'rolled_back_from': current_version.version,
                'rollback_reason': 'manual_rollback'
            }
            
            self.registry.update_model_version(
                name=name,
                version=previous_version.version,
                tags=rollback_tags
            )
            
            logger.info(f"Successfully rolled back model '{name}' in stage '{target_stage}' "
                       f"from v{current_version.version} to v{previous_version.version}")
            
            return {
                'success': True,
                'model_name': name,
                'stage': target_stage,
                'rolled_back_from': current_version.version,
                'rolled_back_to': previous_version.version,
                'rollback_timestamp': rollback_tags['rolled_back_at']
            }
            
        except Exception as e:
            error_msg = f"Failed to rollback model '{name}' in stage '{target_stage}': {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def get_promotion_history(self, name: str, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get promotion history for a model or specific version.
        
        Args:
            name: Model name
            version: Optional specific version
            
        Returns:
            List of promotion events
        """
        try:
            # Get model versions
            if version:
                versions = [self.registry.get_model_version(name, version)]
                versions = [v for v in versions if v is not None]
            else:
                versions = self.registry.search_model_versions(f"name='{name}'")
            
            promotion_history = []
            
            for model_version in versions:
                # Extract promotion information from tags
                tags = model_version.tags or {}
                
                if 'promoted_at' in tags:
                    promotion_history.append({
                        'version': model_version.version,
                        'promoted_at': tags.get('promoted_at'),
                        'promoted_from': tags.get('promoted_from'),
                        'promoted_to': tags.get('promoted_to'),
                        'promotion_type': tags.get('promotion_type'),
                        'current_stage': model_version.current_stage,
                    })
                
                if 'rolled_back_at' in tags:
                    promotion_history.append({
                        'version': model_version.version,
                        'rolled_back_at': tags.get('rolled_back_at'),
                        'rolled_back_from': tags.get('rolled_back_from'),
                        'rollback_reason': tags.get('rollback_reason'),
                        'current_stage': model_version.current_stage,
                    })
            
            # Sort by timestamp
            promotion_history.sort(
                key=lambda x: x.get('promoted_at') or x.get('rolled_back_at') or '',
                reverse=True
            )
            
            return promotion_history
            
        except Exception as e:
            logger.error(f"Failed to get promotion history for '{name}': {e}")
            return []
    
    def get_stage_summary(self, name: str) -> Dict[str, Any]:
        """
        Get summary of model versions across all stages.
        
        Args:
            name: Model name
            
        Returns:
            Stage summary dictionary
        """
        try:
            stages = ['staging', 'production', 'archived', 'none']
            stage_summary = {}
            
            for stage in stages:
                latest_version = self.registry.get_latest_model_version(name, stage)
                if latest_version:
                    stage_summary[stage] = {
                        'version': latest_version.version,
                        'creation_timestamp': latest_version.creation_timestamp,
                        'last_updated_timestamp': latest_version.last_updated_timestamp,
                        'description': latest_version.description,
                        'run_id': latest_version.run_id,
                        'status': latest_version.status,
                    }
                else:
                    stage_summary[stage] = None
            
            return {
                'model_name': name,
                'stages': stage_summary,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Failed to get stage summary for '{name}': {e}")
            return {'error': str(e)}


# Global promotion workflow instance
_promotion_workflow: Optional[ModelPromotionWorkflow] = None


def get_promotion_workflow() -> ModelPromotionWorkflow:
    """Get or create global model promotion workflow instance."""
    global _promotion_workflow
    
    if _promotion_workflow is None:
        _promotion_workflow = ModelPromotionWorkflow()
    
    return _promotion_workflow