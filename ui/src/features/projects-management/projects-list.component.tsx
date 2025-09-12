/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ActionMenu, Flex, Item, Menu, Text } from 'packages/ui';

import styles from './projects-list.module.scss';

interface Project {
    name: string;
    id: string;
}
interface ProjectListProps {
    menuWidth?: string;
    projects: Project[];
}

export const ProjectsList = ({ menuWidth = '100%', projects }: ProjectListProps) => {
    return (
        <Menu UNSAFE_className={styles.projectMenu} width={menuWidth} items={projects}>
            {(item) => (
                <Item key={item.id} textValue={item.name}>
                    <Text>
                        <Flex justifyContent='space-between' alignItems='center' marginX={'size-200'}>
                            {item.name}
                            <ActionMenu isQuiet>
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
