/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, useState } from 'react';

import { ActionMenu, Flex, Item, TextField } from '@geti/ui';

import styles from './projects-list.module.scss';

interface Project {
    name: string;
    id: string;
}
interface ProjectListProps {
    projects: Project[];
    projectInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

interface ProjectEditionProps {
    onBlur: () => void;
    name: string;
}

const ProjectEdition = ({ name, onBlur }: ProjectEditionProps) => {
    const [newName, setNewName] = useState<string>(name);

    return (
        <TextField
            isQuiet
            ref={(ref) => {
                ref?.select();
            }}
            value={newName}
            onBlur={onBlur}
            onChange={setNewName}
        />
    );
};

const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
};

interface ProjectActionsProps {
    onAction: (key: Key) => void;
}

const ProjectActions = ({ onAction }: ProjectActionsProps) => {
    return (
        <ActionMenu isQuiet onAction={onAction} aria-label={'Project actions'} UNSAFE_className={styles.actionMenu}>
            {[PROJECT_ACTIONS.RENAME, PROJECT_ACTIONS.DELETE].map((action) => (
                <Item key={action}>{action}</Item>
            ))}
        </ActionMenu>
    );
};

interface ProjectListItemProps {
    project: Project;
    isInEditMode: boolean;
    onBlur: () => void;
    onAction: (key: Key) => void;
}

const ProjectListItem = ({ project, isInEditMode, onAction, onBlur }: ProjectListItemProps) => {
    return (
        <li className={styles.projectListItem}>
            <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                {isInEditMode ? <ProjectEdition name={project.name} onBlur={onBlur} /> : project.name}
                <ProjectActions onAction={onAction} />
            </Flex>
        </li>
    );
};

export const ProjectsList = ({ projects, setProjectInEdition, projectInEdition }: ProjectListProps) => {
    const isInEditionMode = (projectId: string) => {
        return projectInEdition === projectId;
    };

    const handleBlur = () => {
        setProjectInEdition(null);
    };

    const handleAction = (projectId: string) => (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            setProjectInEdition(projectId);
        }
    };

    return (
        <ul className={styles.projectList}>
            {projects.map((project) => (
                <ProjectListItem
                    key={project.id}
                    project={project}
                    onAction={handleAction(project.id)}
                    onBlur={handleBlur}
                    isInEditMode={isInEditionMode(project.id)}
                />
            ))}
        </ul>
    );
};
