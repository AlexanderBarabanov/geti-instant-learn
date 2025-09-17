/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Key, useEffect, useRef } from 'react';

import { ActionMenu, Flex, Item, Menu, Text, TextField, TextFieldRef } from 'packages/ui';

import styles from './projects-list.module.scss';

interface Project {
    name: string;
    id: string;
}
interface ProjectListProps {
    menuWidth?: string;
    projects: Project[];
    projectInEdition: string | null;
    setProjectInEdition: (projectId: string | null) => void;
}

const ProjectEdition = ({ name, onBlur }: { name: string; onBlur: () => void }) => {
    const inputRef = useRef<TextFieldRef<HTMLInputElement>>(null);

    useEffect(() => {
        inputRef.current?.select();
    }, []);

    return <TextField ref={inputRef} value={name} onBlur={onBlur} onChange={(e) => console.log(e.target.value)} />;
};

export const ProjectsList = ({
    menuWidth = '100%',
    projects,
    setProjectInEdition,
    projectInEdition,
}: ProjectListProps) => {
    const isInEditionMode = (projectId: string) => projectInEdition === projectId;

    const looseFocus = () => {
        setProjectInEdition(null);
    };

    const onActionHandler = (key: Key, id: string) => {
        if (key === 'Rename') {
            setProjectInEdition(id);
        }
    };

    return (
        <Menu UNSAFE_className={styles.projectMenu} width={menuWidth} items={projects}>
            {(item) => (
                <Item key={item.id} textValue={item.name}>
                    <Text>
                        <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                            {isInEditionMode(item.id) ? (
                                <ProjectEdition name={item.name} onBlur={looseFocus} />
                            ) : (
                                item.name
                            )}
                            <ActionMenu isQuiet onAction={(key) => onActionHandler(key, item.id)}>
                                <Item>Rename</Item>
                                <Item>Delete</Item>
                            </ActionMenu>
                        </Flex>
                    </Text>
                </Item>
            )}
        </Menu>
    );
};
